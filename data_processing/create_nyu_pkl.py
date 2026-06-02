import argparse
import gzip
import json
import pickle

import mat73
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Pack NYUv2 depth/snorm data into a single pkl file.")
    parser.add_argument("--root", required=True, help="Path to the NYUv2 dataset root directory")
    args = parser.parse_args()

    root_dir = args.root.rstrip("/") + "/"

    # load original dicts
    nyuv2_dict = mat73.loadmat(root_dir + "nyu_depth_v2_labeled.mat")
    with gzip.GzipFile(root_dir + "all_normals.pklz", "r") as file:
        snorm_dict = pickle.load(file)

    # get split data
    train_json = json.load(open(root_dir + "train_SN40.json"))
    test_json = json.load(open(root_dir + "test_SN40.json"))
    train_split = [int(_ins["img"].split("_")[0]) - 1 for _ins in train_json]
    test_split = [int(_ins["img"].split("_")[0]) - 1 for _ins in test_json]

    # Save dictionary
    save_dict = {
        "depths": np.transpose(nyuv2_dict["rawDepths"], (2, 0, 1)),
        "images": np.transpose(nyuv2_dict["images"], (3, 2, 0, 1)),
        "snorms": np.transpose(snorm_dict["all_normals"], (0, 3, 1, 2)),
        "scene_types": nyuv2_dict["sceneTypes"],
        "train_indices": np.array(train_split),
        "test_indices": np.array(test_split),
    }

    save_path = root_dir + "nyuv2_snorm_all.pkl"
    print(f"Saving combined pkl file at {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    main()
