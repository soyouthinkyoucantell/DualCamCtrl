import multiprocessing as mp
from tqdm import tqdm

import pdb
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from sympy import parsing
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
import random
import torch.nn.functional as F
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
# import dataset
from torch.utils.data import Dataset
from .graphics_utils import focal2fov
from .camera_utils import CameraInfo, Pose
import torchvision
import sys
import os
# import depth_pro
# depth_estimator, transform = depth_pro.create_model_and_transforms()

import json
import os
import os.path as osp
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch


from packaging import version as pver

meta_json = '/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/camera_data_paths.json'


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + \
        0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + \
        0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / \
        directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3,
                              :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(
        rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    # B, V, HW, 3
    # print(f"rays_o shape: {rays_o.shape}, rays_d shape: {rays_d.shape}")
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    # print(f"rays_dxo shape: {rays_dxo.shape}")
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(
        B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class BaseParser(object):
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: Optional[int] = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        self.image_names: List[str] = []  # (num_images,)
        self.image_paths: List[str] = []  # (num_images,)
        self.camtoworlds: np.ndarray = np.zeros(
            (0, 4, 4))  # (num_images, 4, 4)
        self.camera_ids: List[int] = []  # (num_images,)
        self.Ks_dict: Dict[int, np.ndarray] = {}  # Dict of camera_id -> K
        # Dict of camera_id -> params
        self.params_dict: Dict[int, np.ndarray] = {}
        self.imsize_dict: Dict[
            int, Tuple[int, int]
        ] = {}  # Dict of camera_id -> (width, height)
        self.points: np.ndarray = np.zeros((0, 3))  # (num_points, 3)
        self.points_err: np.ndarray = np.zeros((0,))  # (num_points,)
        self.points_rgb: np.ndarray = np.zeros((0, 3))  # (num_points, 3)
        # Dict of image_name -> (M,)
        self.point_indices: Dict[str, np.ndarray] = {}
        self.transform: np.ndarray = np.zeros((4, 4))  # (4, 4)

        # Dict of camera_id -> (H, W)
        self.mapx_dict: Dict[int, np.ndarray] = {}
        # Dict of camera_id -> (H, W)
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, Tuple[int, int, int, int]] = (
            dict()
        )  # Dict of camera_id -> (x, y, w, h)
        self.scene_scale: float = 1.0


class ReconfusionParser(BaseParser):
    def __init__(self, data_dir: str, factor: int = 1, normalize: bool = False):
        super().__init__(data_dir, factor, normalize, test_every=None)

        with open(osp.join(data_dir, "transforms.json")) as f:
            metadata = json.load(f)
        # print(f"metadata: {metadata.keys()}")
        image_names, image_paths, camtoworlds = [], [], []
        for frame in metadata["frames"]:
            # pdb.set_trace()
            if frame["file_path"] is None:
                image_path = image_name = None
            else:
                image_path = osp.join(data_dir, frame["file_path"])
                # TODO
                image_path = image_path.replace('images', 'images_4')
                image_name = osp.basename(image_path)
            image_paths.append(image_path)
            image_names.append(image_name)
            camtoworld = np.array(frame["transform_matrix"])
            if "applied_transform" in metadata:
                applied_transform = np.concatenate(
                    [metadata["applied_transform"], [[0, 0, 0, 1]]], axis=0
                )
                camtoworld = np.linalg.inv(applied_transform) @ camtoworld
            camtoworlds.append(camtoworld)
        camtoworlds = np.array(camtoworlds)
        camtoworlds[:, :, [1, 2]] *= -1

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            self.transform = T1
        else:
            self.transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = list(range(len(image_paths)))

        self.Ks_dict = {
            i: np.array(
                [
                    [
                        metadata.get("fl_x", frame.get("fl_x", None)),
                        0.0,
                        metadata.get("cx", frame.get("cx", None)),
                    ],
                    [
                        0.0,
                        metadata.get("fl_y", frame.get("fl_y", None)),
                        metadata.get("cy", frame.get("cy", None)),
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )
            for i, frame in enumerate(metadata["frames"])
        }
        self.imsize_dict = {
            i: (
                metadata.get("w", frame.get("w", None)),
                metadata.get("h", frame.get("h", None)),
            )
            for i, frame in enumerate(metadata["frames"])
        }
        # When num_input_frames is None, use all frames for both training and
        # testing.
        # self.splits_per_num_input_frames[None] = {
        #     "train_ids": list(range(len(image_paths))),
        #     "test_ids": list(range(len(image_paths))),
        # }

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        self.bounds = None
        if osp.exists(osp.join(data_dir, "bounds.npy")):
            self.bounds = np.load(osp.join(data_dir, "bounds.npy"))
            scaling = np.linalg.norm(self.transform[0, :3])
            self.bounds = self.bounds / scaling
