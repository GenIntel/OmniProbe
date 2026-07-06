import pytest
import torch
from omegaconf import OmegaConf

from omniprobe.utils.eval_helpers import resolve_correspondence_image_size


class PatchGridBackbone(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images):
        batch_size = images.shape[0]
        h = images.shape[-2] // int(self.patch_size)
        w = images.shape[-1] // int(self.patch_size)
        return torch.ones(batch_size, 1, h, w, device=images.device)


def _cfg(**values):
    defaults = {
        "image_size": 800,
        "image_size_policy": "nearest_multiple",
        "fixed_patched_size": False,
        "num_patches": 60,
    }
    defaults.update(values)
    return OmegaConf.create(defaults)


def test_correspondence_image_size_exact_multiple_stays_unchanged():
    info = resolve_correspondence_image_size(_cfg(image_size=800), PatchGridBackbone(16))
    assert info["requested_image_size"] == 800
    assert info["effective_image_size"] == 800
    assert info["expected_grid_hw"] == (50, 50)


def test_correspondence_image_size_rounds_to_nearest_patch_multiple():
    info = resolve_correspondence_image_size(_cfg(image_size=800), PatchGridBackbone(14))
    assert info["effective_image_size"] == 798
    assert info["expected_grid_hw"] == (57, 57)


def test_correspondence_image_size_tie_rounds_down():
    info = resolve_correspondence_image_size(_cfg(image_size=21), PatchGridBackbone(14))
    assert info["effective_image_size"] == 14


def test_correspondence_image_size_fixed_patch_grid_verifies_forward_shape():
    info = resolve_correspondence_image_size(
        _cfg(image_size=800, fixed_patched_size=True, num_patches=60),
        PatchGridBackbone(14),
    )
    assert info["effective_image_size"] == 840
    assert info["expected_grid_hw"] == (60, 60)
    assert info["verified_grid_hw"] == (60, 60)
    assert info["image_size_policy"] == "fixed_patch_grid"


def test_correspondence_image_size_rejects_invalid_patch_size():
    with pytest.raises(ValueError, match="patch_size"):
        resolve_correspondence_image_size(_cfg(), PatchGridBackbone(0))


def test_correspondence_image_size_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported image_size_policy"):
        resolve_correspondence_image_size(
            _cfg(image_size_policy="ceil"),
            PatchGridBackbone(14),
        )
