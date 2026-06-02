# Backbone Models

All backbone configs live under `configs/backbone/<name>.yaml`. The config name is the value you pass to `backbone=<name>` in any training or evaluation script.

The table lists every available config, its weight source(s), and which output modes it supports (`dense` = per-patch feature map, `cls` = CLS / summary token). All probing pipelines use `dense`; `cls` is available where marked. Source labels: `torch.hub`, `open_clip`, `timm`, `HF Hub`, `ckpt` (local checkpoint), `vendored` (code under `omniprobe/models/vendor/`).

| Model | Source(s) | dense | cls |
|-------|-----------|:-----:|:---:|
| `dino_b16` | torch.hub | ✓ | ✓ |
| `dino_b8` | torch.hub | ✓ | ✓ |
| `dino_s16` | torch.hub | ✓ | ✓ |
| `dinov2_s14` | torch.hub | ✓ | ✓ |
| `dinov2_b14` | torch.hub | ✓ | ✓ |
| `dinov2_b14_reg` | torch.hub | ✓ | ✓ |
| `dinov2_l14` | torch.hub | ✓ | ✓ |
| `dinov2_g14` | torch.hub | ✓ | ✓ |
| `dinov2_b14_diy_sc` | torch.hub | ✓ | — |
| `dinov3_vits16` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vits16plus` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vitb16` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vitl16` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vitl16_sat` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vitl16plus` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vith16plus` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vit7b16` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_vit7b16_sat` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_convnext_tiny` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_convnext_small` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_convnext_base` | torch.hub, ckpt | ✓ | ✓ |
| `dinov3_convnext_large` | torch.hub, ckpt | ✓ | ✓ |
| `c_radio_3_b` | torch.hub | ✓ | ✓ |
| `c_radio_3_l` | torch.hub | ✓ | ✓ |
| `c_radio_3_h` | torch.hub | ✓ | ✓ |
| `c_radio_3_g` | torch.hub | ✓ | ✓ |
| `c_radio_4_h` | torch.hub | ✓ | ✓ |
| `c_radio_4_so400m` | torch.hub | ✓ | ✓ |
| `dune_vits14_448` | torch.hub | ✓ | ✓ |
| `dune_vitb14_336` | torch.hub | ✓ | ✓ |
| `dune_vitb14` | torch.hub | ✓ | ✓ |
| `dune_vitb14_448_paper` | torch.hub | ✓ | ✓ |
| `vjepa2_large` | torch.hub | ✓ | — |
| `vjepa2_huge` | torch.hub | ✓ | — |
| `vjepa2_giant` | torch.hub | ✓ | — |
| `vjepa2_giant_384` | torch.hub | ✓ | — |
| `vjepa2_1_base` | torch.hub, ckpt | ✓ | — |
| `vjepa2_1_large` | torch.hub, ckpt | ✓ | — |
| `clip_b16` | open_clip | ✓ | ✓ |
| `clip_b16_laion` | open_clip | ✓ | ✓ |
| `clip_l14` | open_clip | ✓ | ✓ |
| `clip_h14` | open_clip | ✓ | ✓ |
| `openclip_vitb16_laion2b` | open_clip, ckpt | ✓ | ✓ |
| `openclip_vitb16_datacomp` | open_clip, ckpt | ✓ | ✓ |
| `openclip_vitl14_laion2b` | open_clip, ckpt | ✓ | ✓ |
| `openclip_vitl14_datacomp` | open_clip, ckpt | ✓ | ✓ |
| `clip_convnext` | open_clip | ✓ | — |
| `clip_convnext_augreg` | open_clip | ✓ | — |
| `convnext_fcmae` | timm | ✓ | — |
| `convnext_in22k` | timm | ✓ | — |
| `deit3_b16` | timm | ✓ | ✓ |
| `deit3_l16` | timm | ✓ | ✓ |
| `ibot_s16` | ckpt | ✓ | ✓ |
| `ibot_b16` | ckpt | ✓ | ✓ |
| `ibot_b16_in22k` | ckpt | ✓ | ✓ |
| `ibot_l16` | ckpt | ✓ | ✓ |
| `ibot_l16_in22k` | ckpt | ✓ | ✓ |
| `mae_b16` | HF Hub | ✓ | ✓ |
| `mae_l16` | HF Hub | ✓ | ✓ |
| `mae_h14` | HF Hub | ✓ | ✓ |
| `siglip_b16` | timm | ✓ | — |
| `siglip_l16` | timm | ✓ | — |
| `sam_base` | ckpt | ✓ | — |
| `sam_large` | ckpt | ✓ | — |
| `sam_huge` | ckpt | ✓ | — |
| `metaclip2_vits16` | vendored, ckpt | ✓ | ✓ |
| `metaclip2_vitb16` | vendored, ckpt | ✓ | ✓ |
| `metaclip2_vitl14` | vendored, ckpt | ✓ | ✓ |
| `pixio_vitb16` | vendored, ckpt | ✓ | ✓ |
| `pixio_vitl16` | vendored, ckpt | ✓ | ✓ |
| `pixio_vith16` | vendored, ckpt | ✓ | ✓ |
| `pixio_vit1b16` | vendored, ckpt | ✓ | ✓ |
| `pixio_vit5b16` | vendored, ckpt | ✓ | ✓ |
| `perception_t16_512` | vendored, ckpt | ✓ | — |
| `perception_s16_512` | vendored, ckpt | ✓ | — |
| `perception_b16_512` | vendored, ckpt | ✓ | — |
| `perception_l14_448` | vendored, ckpt | ✓ | — |
| `perception_g14_448` | vendored, ckpt | ✓ | — |
| `crocov2` | vendored, ckpt | ✓ | — |
| `ijepa_vith16_448` | vendored, ckpt | ✓ | — |
| `vggt` | vendored, ckpt | ✓ | — |
| `vggt_dino` | vendored, ckpt | ✓ | — |
| `midas_l16` | torch.hub | ✓ | — |
| `dift_sd21` | HF Hub | ✓ | — |
| `dift_sd15` | HF Hub | ✓ | — |
| `qwen2_vl_3b` | HF Hub | ✓ | — |
| `qwen2_vl_7b` | HF Hub | ✓ | — |
| `qwen2_5_vl_7b` | HF Hub | ✓ | — |
| `qwen3_vl_4b` | HF Hub | ✓ | — |
| `qwen3_vl_8b` | HF Hub | ✓ | — |
| `internvl3_5_8b` | HF Hub | ✓ | — |
| `llava_ov_7b` | HF Hub | ✓ | — |

`dift_*` requires the `ldm` package and the LVLM configs (`qwen*`, `internvl*`, `llava*`) require recent `transformers` VL bindings — neither is in the default env.

## Environment variables for checkpoint paths

| Variable | Used by | Default |
|----------|---------|---------|
| `CROCO_CKPT` | crocov2 | `checkpoints/croco/CroCo_V2_ViTBase_BaseDecoder.pth` |
| `IJEPA_CKPT` | ijepa_vith16_448 | `checkpoints/ijepa/IN1K-vit.h.16-448px-300e.pth.tar` |
| `VGGT_CKPT` | vggt, vggt_dino | `checkpoints/vggt/vggt_1B_commercial.pt` |
| `DIFT_SD15_MODEL_ID` | dift_sd15 | `runwayml/stable-diffusion-v1-5` |
| `TORCH_HOME` | all hub models | set to scratch dir to avoid filling `$HOME` |

A checkpoint env var is resolved relative to the checkpoint root — the repo-local `checkpoints/` directory, or `$OMNIPROBE_PRETRAINED_MODELS` if set — unless you give an absolute path. Set e.g. `CROCO_CKPT=croco/foo.pth` (not `checkpoints/croco/foo.pth`) to avoid a doubled prefix.
