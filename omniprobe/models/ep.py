import torch
from torch import nn


class EfficientProbing(nn.Module):
    def __init__(self, dim: int, num_queries: int = 8, num_heads: int = 1):
        super().__init__()
        assert dim % num_queries == 0, f"dim {dim} must be divisible by num_queries {num_queries}"
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.scale = (dim // num_heads) ** -0.5
        self.v = nn.Linear(dim, dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)

        q = cls.reshape(batch_size, self.num_queries, self.num_heads, dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = x.reshape(batch_size, num_tokens, self.num_heads, dim // self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = self.v(x).reshape(batch_size, num_tokens, self.num_queries, dim // self.num_queries)
        v = v.permute(0, 2, 1, 3)

        attn = (q * self.scale @ k.transpose(-2, -1)).softmax(dim=-1)
        out = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        return out.view(batch_size, dim)
