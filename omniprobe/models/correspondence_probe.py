"""
Linear probe for correspondence feature transformation.

This module provides a simple linear probe that transforms backbone features
before computing correspondences. The probe is trained on the train split
of SPair-71k to improve feature matching performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityProbe(nn.Module):
    """
    Identity probe that passes features through unchanged.
    Used as a baseline to verify that the evaluation matches nearest-neighbor results.
    """
    
    def __init__(self, input_dim: int = None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        # Add a dummy parameter so the module has parameters (for optimizer compatibility)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass features through unchanged."""
        return x


class LinearCorrespondenceProbe(nn.Module):
    """
    A linear probe that transforms features for correspondence matching.
    
    Takes flattened spatial features and outputs linearly transformed features
    that can be used for nearest neighbor matching.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        bias: bool = True,
        init_mode: str = "identity",
    ):
        """
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features (defaults to input_dim)
            bias: Whether to use bias in the linear layer
            init_mode: Initialization mode ("identity", "random", "xavier")
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_mode = init_mode
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Initialize weights based on mode
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights based on init_mode."""
        if self.init_mode == "identity":
            # Initialize as identity: features pass through unchanged
            nn.init.eye_(self.linear.weight[:min(self.input_dim, self.output_dim), 
                                             :min(self.input_dim, self.output_dim)])
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        elif self.init_mode == "random":
            # Default PyTorch initialization (kaiming uniform)
            nn.init.kaiming_uniform_(self.linear.weight, a=5**0.5)
            if self.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
                bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
                nn.init.uniform_(self.linear.bias, -bound, bound)
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform features through linear probe.
        
        Args:
            x: Input features of shape (B, C, H, W) or (N, C)
            
        Returns:
            Transformed features of same spatial shape
        """
        if x.dim() == 4:
            # (B, C, H, W) -> (B, H, W, C) -> linear -> (B, H, W, C') -> (B, C', H, W)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.linear(x)  # (B, H, W, C')
            x = x.permute(0, 3, 1, 2)  # (B, C', H, W)
        elif x.dim() == 2:
            # (N, C) -> linear -> (N, C')
            x = self.linear(x)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")
            
        return x


class PositiveLinearCorrespondenceProbe(nn.Module):
    """
    A linear probe with strictly positive weights.

    The weight matrix is parameterized in unconstrained space and mapped through
    softplus during the forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        bias: bool = True,
        init_mode: str = "identity",
        min_weight: float = 1e-6,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim
        if min_weight < 0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_mode = init_mode
        self.min_weight = float(min_weight)

        self.raw_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _inverse_softplus(self, target: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(target))

    def _init_weights(self):
        if self.init_mode == "identity":
            target_weight = torch.full(
                (self.output_dim, self.input_dim),
                self.min_weight,
                dtype=self.raw_weight.dtype,
                device=self.raw_weight.device,
            )
            diag_size = min(self.input_dim, self.output_dim)
            diag_index = torch.arange(diag_size, device=self.raw_weight.device)
            target_weight[diag_index, diag_index] = 1.0
            self.raw_weight.data.copy_(self._inverse_softplus(target_weight - self.min_weight))
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init_mode == "random":
            nn.init.normal_(self.raw_weight, mean=0.0, std=0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init_mode == "xavier":
            target_weight = torch.empty_like(self.raw_weight)
            nn.init.xavier_uniform_(target_weight)
            target_weight = target_weight.abs().clamp_min(self.min_weight)
            self.raw_weight.data.copy_(self._inverse_softplus(target_weight - self.min_weight))
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def _weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight) + self.min_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._weight()
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = F.linear(x, weight, self.bias)
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 2:
            x = F.linear(x, weight, self.bias)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")
        return x


class MLPCorrespondenceProbe(nn.Module):
    """
    A 2-layer MLP probe for correspondence feature transformation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        output_dim: int = None,
        bias: bool = True,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer (defaults to input_dim)
            output_dim: Dimension of output features (defaults to input_dim)
            bias: Whether to use bias
            activation: Activation function ("relu", "gelu")
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform features through MLP probe.
        
        Args:
            x: Input features of shape (B, C, H, W) or (N, C)
            
        Returns:
            Transformed features of same spatial shape
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            x = x.permute(0, 3, 1, 2)  # (B, C', H, W)
        elif x.dim() == 2:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")
            
        return x


def build_correspondence_probe(
    probe_type: str,
    input_dim: int,
    output_dim: int = None,
    hidden_dim: int = None,
    init_mode: str = "identity",
    **kwargs
) -> nn.Module:
    """
    Factory function to build correspondence probes.
    
    Args:
        probe_type: Type of probe ("identity", "linear", "positive_linear", "mlp")
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        hidden_dim: Hidden dimension (for MLP)
        init_mode: Weight initialization mode for linear/mlp ("identity", "random", "xavier")
        **kwargs: Additional arguments passed to probe constructor
        
    Returns:
        Probe module
    """
    if probe_type == "identity":
        return IdentityProbe(input_dim=input_dim)
    elif probe_type == "linear":
        return LinearCorrespondenceProbe(input_dim, output_dim, init_mode=init_mode, **kwargs)
    elif probe_type == "positive_linear":
        return PositiveLinearCorrespondenceProbe(input_dim, output_dim, init_mode=init_mode, **kwargs)
    elif probe_type == "mlp":
        return MLPCorrespondenceProbe(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
