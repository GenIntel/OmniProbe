import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as transforms

from .utils import read_image
from omniprobe.utils.eval_helpers import resolve_mean_std
from omniprobe.utils.paths import require_env_path


class ScanNetPairsDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | None = None, image_mean: str = "perception"):
        super().__init__()

        # Some defaults for consistency.
        self.name = "ScanNet-pairs"
        self.root = root or require_env_path("SCANNET_ROOT", "ScanNet pairs root")
        self.split = "test"
        self.num_views = 2
        mean, std = resolve_mean_std(image_mean)

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # parse files for data
        self.instances = self.get_instances(self.root)

        # Print out dataset stats
        print(f"{self.name} | {len(self.instances)} pairs")

    def get_dep(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = np.array(img).astype(np.int32)
                img = torch.tensor(img).float() / 1000.0
                return img[None, :, :]

    def get_instances(self, root_path):
        K_dict = dict(np.load(f"{root_path}/intrinsics.npz"))
        data = np.load(f"{root_path}/test.npz")["name"]
        instances = []

        for i in range(len(data)):
            room_id, seq_id, ins_0, ins_1 = data[i]
            scene_id = f"scene{room_id:04d}_{seq_id:02d}"
            K_i = torch.tensor(K_dict[scene_id]).float()

            instances.append((scene_id, ins_0, ins_1, K_i))

        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        s_id, ins_0, ins_1, K = self.instances[index]

        # paths
        rgb_path_0 = os.path.join(self.root, s_id, f"color/{ins_0}.jpg")
        rgb_path_1 = os.path.join(self.root, s_id, f"color/{ins_1}.jpg")
        dep_path_0 = os.path.join(self.root, s_id, f"depth/{ins_0}.png")
        dep_path_1 = os.path.join(self.root, s_id, f"depth/{ins_1}.png")

        # get rgb
        rgb_0 = read_image(rgb_path_0, exif_transpose=False)
        rgb_1 = read_image(rgb_path_1, exif_transpose=False)
        rgb_0 = self.rgb_transform(rgb_0)
        rgb_1 = self.rgb_transform(rgb_1)

        # get depths
        dep_0 = self.get_dep(dep_path_0)
        dep_1 = self.get_dep(dep_path_1)

        # get poses
        pose_path_0 = os.path.join(self.root, s_id, f"pose/{ins_0}.txt")
        pose_path_1 = os.path.join(self.root, s_id, f"pose/{ins_1}.txt")
        Rt_0 = torch.tensor(np.loadtxt(pose_path_0, delimiter=" "))
        Rt_1 = torch.tensor(np.loadtxt(pose_path_1, delimiter=" "))
        Rt_01 = Rt_1.inverse() @ Rt_0

        return {
            "uid": index,
            "class_id": "ScanNet_test",
            "sequence_id": s_id,
            "frame_0": int(ins_0),
            "frame_1": int(ins_1),
            "K": K,
            "rgb_0": rgb_0,
            "rgb_1": rgb_1,
            "depth_0": dep_0,
            "depth_1": dep_1,
            "Rt_0": torch.eye(4).float(),
            "Rt_1": Rt_01.float(),
        }
