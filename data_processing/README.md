# Datasets

Each evaluation task expects a dataset root directory configured via an environment variable (set in your `.env` or shell). The table below summarises all variables; detailed download instructions follow.

| Variable | Task(s) | Section |
|----------|---------|---------|
| `NAVI_ROOT` | correspondence_navi | [NAVI](#navi) |
| `SCANNET_ROOT` | correspondence_scannet | [ScanNet](#scannet-correspondence-test-split) |
| `SPAIR_ROOT` | correspondence_spair | [SPair-71k](#spair-71k) |
| `NYU_ROOT` | depth, snorm | [NYU](#nyu-dataset) |
| `ADE20K_ROOT` | segmentation_ade20k | [ADE20K](#ade20k) |
| `IMAGENET_ROOT` | classification_imagenet_knn, classification_imagenet_linear | [ImageNet](#imagenet) |
| `IMAGENET3D_ROOT` | pose_imagenet3d | [ImageNet3D](#imagenet3d) |
| `TAPVID_DAVIS_ROOT` | tracking_tapvid | [TAP-Vid DAVIS](#tap-vid-davis) |
| `AP10K_ROOT` | correspondence_ap10k | [AP-10K](#ap-10k) |
| `SOCO_ROOT` | correspondence_soco | [SOCO](#soco) |


## NAVI

[NAVI](https://github.com/google/navi) is a multi-view dataset depicting 36 objects in varied scenes and poses, with high-quality meshes and precise image-object alignment. NAVI has very high resolution images, so we recommend downsampling first to avoid slow data loading.

```bash
cd data/
wget http://storage.googleapis.com/gresearch/navi-dataset/navi_v1.tar.gz
tar -xzf navi_v1.tar.gz

# downsample images (writes downsampled_ prefixed copies next to originals)
cd ../data_processing
python resize_navi.py --data-root ../data/navi_v1
```

Set `NAVI_ROOT` to the `navi_v1/` directory.


## ScanNet Correspondence Test Split

[ScanNet](https://www.scan-net.org/) is a large RGB-D video dataset of indoor scenes. We use the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) test split (1500 image pairs), downloaded from the [LoFTR](https://github.com/zju3dv/LoFTR/) website.

```bash
cd data/

# download the tar file provided by LoFTR
gdown --id 1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd
tar -xvf scannet_test_1500.tar
rm scannet_test_1500.tar

cd scannet_test_1500
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/intrinsics.npz
wget https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/test.npz
```

Set `SCANNET_ROOT` to the `scannet_test_1500/` directory.


## SPair-71k

[SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/) consists of image pairs depicting instances of the same class with keypoint annotations and viewpoint attributes.

```bash
cd data/
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xvf SPair-71k.tar.gz
rm SPair-71k.tar.gz
```

Set `SPAIR_ROOT` to the `SPair-71k/` directory.


## NYU Dataset

The [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) dataset is a standard depth estimation benchmark. We evaluate on the labeled test set using surface normal annotations from [Ladicky et al.](https://inf.ethz.ch/personal/pomarc/pubs/LadickyECCV14.pdf). For training we use the larger set annotated by [Bansal et al.](https://github.com/aayushbansal/MarrRevisited) / [Qi et al.](https://github.com/xjqi/GeoNet) (GeoNet).

**Step 1 — Download GeoNet training data:**

Download `data1.zip` and `data2.zip` from [GeoNet](https://github.com/xjqi/GeoNet):

```bash
cd data/
# Download data1.zip and data2.zip from https://github.com/xjqi/GeoNet
unzip data1.zip
unzip data2.zip
mkdir nyu_geonet
mv data1/* nyu_geonet/
mv data2/* nyu_geonet/
rmdir data1 data2
```

**Step 2 — Download test set and surface normal annotations:**

```bash
cd data/
mkdir nyuv2 && cd nyuv2

wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip

unzip nyuv2_surfacenormal_metadata.zip
mv surfacenormal_metadata/* .
rm nyuv2_surfacenormal_metadata.zip
rmdir surfacenormal_metadata
```

**Step 3 — Pack into a single pkl:**

```bash
cd ../../data_processing
python create_nyu_pkl.py --root ../data/nyuv2
```

Set `NYU_ROOT` to the parent directory containing both `nyu_geonet/` and `nyuv2/`.


## ADE20K

[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) (MIT Scene Parsing Benchmark) is used for linear-probe semantic segmentation evaluation. Download from the [official site](http://sceneparsing.csail.mit.edu/) or via:

```bash
cd data/
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
```

Expected structure:
```
ADEChallengeData2016/
  images/training/
  images/validation/
  annotations/training/
  annotations/validation/
```

Set `ADE20K_ROOT` to the `ADEChallengeData2016/` directory.


## ImageNet

[ImageNet](https://www.image-net.org/) (ILSVRC 2012) is used for kNN and linear-probe classification evaluation. Download from the [official site](https://www.image-net.org/download.php) (requires registration).

The code loads splits via `torchvision.datasets.ImageFolder` from `{IMAGENET_ROOT}/{split}/`. The default split names in the task configs are `ordered_train` and `ordered`; override via CLI if your layout differs (e.g. `task.train_split=train task.val_split=val`).

Set `IMAGENET_ROOT` to the directory containing the split subdirectories.


## ImageNet3D

[ImageNet3D](https://github.com/wufeim/imagenet3d) provides 3D pose annotations for ImageNet images. Follow the download instructions in the [official repository](https://github.com/wufeim/imagenet3d).

Expected structure:
```
ImageNet3D/
  JPEGImages/{class_name}/*.JPEG
  Segmentation/{class_name}/*.png
  PairAnnotation/{split}/*.json
  ImageAnnotation/{class_name}/*.json
```

Set `IMAGENET3D_ROOT` to the `ImageNet3D/` directory.


## TAP-Vid DAVIS

[TAP-Vid](https://github.com/google-deepmind/tapnet) is a benchmark for tracking any point through video. We use the DAVIS subset. Download from the [TAP-Vid data page](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid):

```bash
cd data/
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip
rm tapvid_davis.zip
```

Set `TAPVID_DAVIS_ROOT` to the extracted `.pkl` file path (not a directory) — the code opens it directly as a pickle file.


## AP-10K

[AP-10K](https://github.com/AlexTheBad/AP-10K) is an animal pose estimation dataset with keypoint annotations across species. Follow the download instructions in the [official repository](https://github.com/AlexTheBad/AP-10K).

Expected structure:
```
AP-10K/
  JPEGImages/{family}/{species}/*.jpg
  ImageAnnotation/{family}/{species}/*.json
  PairAnnotation/{split}/*.json
```

Set `AP10K_ROOT` to the `AP-10K/` directory.


## SOCO

[SOCO](https://huggingface.co/datasets/GenIntelLab/SOCO) is a semantic object correspondence dataset with 100 categories (40 images each), per-view keypoint annotations, and within- and cross-category pair annotations (20k intra-category, 20k cross-category, plus a predefined 10k / 10k intra-category train/test split).

The dataset is distributed on the Hugging Face Hub as three zip archives (`Images.zip`, `KeypointAnnotations.zip`, `PairAnnotations.zip`) plus an unpacked `Metadata/` folder. Download the repository and unzip the archives in place:

```bash
huggingface-cli download GenIntelLab/SOCO --repo-type dataset --local-dir data/SOCOv1
cd data/SOCOv1 && for z in *.zip; do unzip -q "$z" && rm "$z"; done
```

Expected structure after extraction:
```
SOCOv1/
  Images/{category}/*.JPEG
  KeypointAnnotations/{category}/*.json
  PairAnnotations/
    intra/{category}/*.json              # all within-category pairs (20k)
    cross/{category}/*.json              # cross-category pairs (20k)
    trainsplits/
      train/{category}/*.json            # training split (10k)
      test/{category}/*.json             # test split (10k)
  Metadata/
    keypoint_taxonomy.json
    filename_mapping.json
```

Set `SOCO_ROOT` to the downloaded `SOCOv1/` directory.

The `pair_subdir` config parameter selects which pair annotations to use (default: `PairAnnotations/intra`).
For linear probe training, `train_pair_subdir` and `test_pair_subdir` select the predefined splits.
