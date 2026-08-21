# Tasks

All task configs live under `configs/task/<name>.yaml`. The config name is the value you pass to `task=<name>` in `python -m omniprobe.run task=<name> backbone=<backbone>`. The live list is `omniprobe --list-tasks`.

The table lists the available tasks, their config(s), and the datasets they run on. Most tasks are a single command (see the [Usage](../README.md#usage) section of the README); tasks that need extra setup are documented in detail below.

| Task | Config(s) | Dataset(s) | Description |
|------|-----------|------------|-------------|
| Correspondence | `correspondence_spair`, `correspondence_soco`, `correspondence_navi`, `correspondence_scannet`, `correspondence_ap10k`, `correspondence_geometric_soco` (+ `_linear_probe` variants for SOCO/SPair) | SPair-71k, SOCO, NAVI, ScanNet, AP-10K | Semantic / geometric keypoint matching from dense features; the `_linear_probe` variants train a probe on top of the frozen features. |
| Tracking | `tracking_tapvid` | TAP-Vid DAVIS | Point tracking through video. |
| Depth | `depth` | NYU, NAVI | Train a depth probe on frozen features. |
| Surface normals | `snorm` | NYU, NAVI | Train a surface-normal probe on frozen features. |
| Segmentation | `segmentation_ade20k` (+ `segmentation_ade20k_eval`) | ADE20K | Semantic segmentation probe (`_eval` runs evaluation only). |
| Pose | `pose_imagenet3d` (+ `pose_imagenet3d_ep`) | ImageNet3D | Object viewpoint / pose estimation. |
| 3D detection | `detection3d_omni3d` (+ `detection3d_omni3d_in`, `_out`, `_full`) | Omni3D (ARKitScenes default; indoor / outdoor / full presets) | Cube R-CNN heads on a frozen backbone; reports AP2D/AP3D. See [3D detection (Omni3D)](#3d-detection-omni3d) below. |
| Classification | `classification_imagenet_knn`, `classification_imagenet_linear` | ImageNet | kNN and linear-probe classification. |

---

## 3D detection (Omni3D)

The 3D detection task trains Cube R-CNN heads on top of a frozen backbone and reports AP2D/AP3D. It needs two things the other tasks don't:

1. **Extra dependencies** — detectron2 and PyTorch3D built from source plus the `detection3d` extra; see the install commands in [Installation](../README.md#installation).
2. **The Omni3D data** — annotations and ARKitScenes images under `OMNI3D_ROOT`; see [data_processing/README.md](../data_processing/README.md#omni3d-3d-detection) for the download and layout.

Training and evaluation on ARKitScenes (the default):

```bash
# full training run (ARKitScenes, frozen backbone, 116k iterations)
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14

# multi-GPU: single node, N processes; solver.ims_per_batch is the TOTAL
# batch size across GPUs (default: num_gpus=4, ims_per_batch=32)
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.system.num_gpus=4

# evaluate a trained checkpoint (no training); visualize_predictions
# additionally renders 3D cuboid overlays + BEV for every 50th test image
# into <run_dir>/inference/iter_final/<dataset>/vis/
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.eval_only=true task.weights=/path/to/model_final.pth \
  task.visualize_predictions=true

# resume an interrupted run: reuse its output directory
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  task.resume=true hydra.run.dir=outputs/<date>/<time>_detection3d_omni3d_dinov2_b14
```

**Other Omni3D datasets.** Preset tasks cover the standard splits — `detection3d_omni3d_in` (SUN RGB-D + Hypersim + ARKitScenes, 38 categories), `detection3d_omni3d_out` (nuScenes + KITTI, 11), and `detection3d_omni3d_full` (all six, 50); run them exactly like the default task once the corresponding datasets are downloaded. For custom combinations, override the dataset lists and let the category set resolve from a preset name (`omni3d`, `omni3d_in`, `omni3d_out`, or any split name):

```bash
python -m omniprobe.run task=detection3d_omni3d backbone=dinov2_b14 \
  'task.datasets.train=[KITTI_train,KITTI_val]' 'task.datasets.test=[KITTI_test]' \
  task.datasets.category_names=KITTI_test task.datasets.num_classes=null
```

When evaluating a checkpoint, keep `category_names`/`num_classes` matching the training run — the detection heads are sized for those categories. The default solver follows the frozen-backbone ARKitScenes recipe (AdamW 1e-3, batch 32, 116k iterations); the upstream Cube R-CNN recipe used larger batches for the bigger splits (128 indoor, 192 full), so expect to tune batch/lr/iterations there.
