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
    def __init__(self, data_dir: str, normalize: bool = False):
        super().__init__(data_dir, 1, normalize, test_every=None)

        def get_num(p):
            return p.split("_")[-1].removesuffix(".json")

        with open(osp.join(data_dir, "transforms.json")) as f:
            metadata = json.load(f)

        image_names, image_paths, camtoworlds = [], [], []
        for frame in metadata["frames"]:
            if frame["file_path"] is None:
                image_path = image_name = None
            else:
                image_path = osp.join(data_dir, frame["file_path"])
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


class ScenesDataset(Dataset):
    """A simple dataset class."""

    def __init__(
        self,
        relative_pose=True,
        split='train',
        ratio=0.99,
        patch_size: Optional[List[int]] = [832, 480],  # W H
    ):
        self.split = split
        with open(meta_json, 'r') as f:
            meta_paths = f.readlines()
        self.meta_paths = [p.strip() for p in meta_paths]
        assert ratio <= 1.0, f"Ratio {ratio} should be less than or equal to 1.0"

        total_scenes = len(self.meta_paths)
        # shuffle
        random.shuffle(self.meta_paths)
        if self.split == 'train':
            num_scenes = int(len(self.meta_paths) * ratio)
            if num_scenes == total_scenes:
                num_scenes = total_scenes - 1
            self.meta_paths = self.meta_paths[:num_scenes]
            print(
                f"Using {num_scenes} scenes for training, total {total_scenes} scenes.")
        elif self.split == 'test':
            num_scenes = int(len(self.meta_paths) * ratio)
            if num_scenes == 0:
                num_scenes = 1
            self.meta_paths = self.meta_paths[-num_scenes:]
            print(
                f"Using {num_scenes} scenes for testing, total {total_scenes} scenes.")
        # random permute the meta_paths
        # random.shuffle(self.meta_paths)
        self.patch_size = patch_size
        self.relative_pose = relative_pose

    def get_relative_pose(self, cam_params):
        # Always zero_init the first camera pose
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + \
            [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(
        self,
        idx,
    ):
        scene_num = len(self.meta_paths)
        idx = (idx+scene_num) % scene_num
        meta_path = self.meta_paths[idx]
        try:
            # print(f"Loading scene {idx} from {meta_path}")
            parsing_file_path = os.path.dirname(meta_path)
            # print(f"Parsing file path: {parsing_file_path}")
            parser = ReconfusionParser(parsing_file_path, normalize=True)

            total_frames = len(parser.image_names)
            if total_frames < 21:
                raise ValueError(
                    f"Total frames {total_frames} is less than 21, cannot sample trajectory.")

            # possible_lengths = list(range(21, 81+1, 4))
            # possible_lengths = [13:65]
            possible_lengths = list(range(13, 65+1, 4))
            sample_steps = [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]

            # print(f"Randoming sample step and length for scene {idx}...")
            if self.split == 'train':
                required_frames = 0x3f3f3f
                max_trys = 100
                _try_times = 0
                chosen_length = np.random.choice(possible_lengths)
                while required_frames > total_frames:
                    _try_times += 1
                    if _try_times > max_trys:
                        sample_step = 1
                        indices = np.arange(
                            0, (41 - 1) * sample_step + 1, sample_step)
                        break
                    sample_step = np.random.choice(sample_steps)
                    required_frames = (chosen_length - 1) * sample_step + 1
                    # print(
                    #     f"Required frames: {required_frames}, Total frames: {total_frames}, Sample step: {sample_step}")
                start = np.random.randint(
                    0, total_frames - required_frames + 1)
                indices = np.arange(
                    start, start + required_frames, sample_step)
            else:
                # 验证阶段默认固定长度和步长
                sample_step = 4
                indices = np.arange(0, (61 - 1) * sample_step + 1, sample_step)

            # print(f"Generated trajectory...")
            # TODO
            trajectory = self.generate_fixed_step_trajectory(parser, indices)
            # print(f"loading images into memory for scene {idx}...")
            data_list = self.load_image_into_memory(
                parser, indices, trajectory)
            # print(
            #     f"K,w2c shape {data_list[0]['cam_info'].K.shape, data_list[0]['cam_info'].w2c.shape}")
            # print(f"Loaded {len(data_list)} frames for scene {idx}.")
            cam_params = [Pose(K=data["cam_info"].K, w2c=data["cam_info"].w2c,
                               image_name=data["cam_info"].image_name,
                               image_path=data["cam_info"].image_path,
                               width=data["cam_info"].width,
                               height=data["cam_info"].height)
                          for data in data_list]

            intrinsics = np.asarray([[cam_param.fx,
                                    cam_param.fy,
                                    cam_param.cx,
                                    cam_param.cy]
                                    for cam_param in cam_params], dtype=np.float32)
            intrinsics = torch.as_tensor(
                intrinsics)[None]                  # [1, n_frame, 4]
            if self.relative_pose:
                c2w_poses = self.get_relative_pose(cam_params)
            else:
                c2w_poses = np.array(
                    [cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
            # [1, n_frame, 4, 4]
            # print(f"Generating plucker embedding for scene {idx}...")
            c2w = torch.as_tensor(c2w_poses)[None]
            plucker_embedding = ray_condition(intrinsics, c2w,
                                              self.patch_size[1], self.patch_size[0],
                                              device='cpu',)[0].permute(0, 3, 1, 2).contiguous()

            data_dict = {
                'images': torch.stack([data["image"].permute(2, 0, 1) for data in data_list], dim=0),
                # Temporary using image as control
                'control': torch.zeros_like(
                    torch.stack([data["image"].permute(2, 0, 1) for data in data_list], dim=0)),
                'camera_infos': plucker_embedding.permute(1, 0, 2, 3)
            }
            if self.split == 'train':
                # Number of frames
                num_frames = len(data_list)

                # Initialize valid mask (first frame is always valid, subsequent frames have 0.2 probability of being active)
                valid_mask = torch.ones(num_frames, dtype=torch.bool)
                possibility_deactivate_list = [0.02, 0.05, 0.1]
                deactivate_prob = random.choice(
                    possibility_deactivate_list)
                valid_mask[1:] = torch.rand(num_frames - 1) < deactivate_prob

                # Create the 'extra_images' and 'extra_image_frame_index' based on the valid mask
                # Select valid images based on the mask
                extra_images = data_dict['images'][valid_mask]
                extra_image_frame_index = torch.nonzero(
                    valid_mask, as_tuple=False).squeeze()  # Get indices of valid frames

                # Remove the first frame from extra_images and extra_image_frame_index
                extra_images = extra_images[1:]  # Skip the first frame
                # Skip the first frame's index
                extra_image_frame_index = extra_image_frame_index[1:]

                # Add the 'extra_images' and 'extra_image_frame_index' to the data_dict
                data_dict['extra_images'] = extra_images
                data_dict['extra_image_frame_index'] = extra_image_frame_index

            elif self.split == 'test':
                # Number of frames
                num_frames = len(data_list)

                # Initialize valid mask (every 5th frame is active, others are inactive)
                valid_mask = torch.zeros(num_frames, dtype=torch.bool)
                # Every
                frame_deacivate = 20
                valid_mask[frame_deacivate-1::frame_deacivate] = True

                # Create the 'extra_images' and 'extra_image_frame_index' based on the valid mask
                # Select valid images based on the mask
                extra_images = data_dict['images'][valid_mask]
                extra_image_frame_index = torch.nonzero(
                    valid_mask, as_tuple=False).squeeze()  # Get indices of valid frames

                # Remove the first frame from extra_images and extra_image_frame_index
                extra_images = extra_images[1:]  # Skip the first frame
                # Skip the first frame's index
                extra_image_frame_index = extra_image_frame_index[1:]

                # Add the 'extra_images' and 'extra_image_frame_index' to the data_dict
                data_dict['extra_images'] = extra_images
                data_dict['extra_image_frame_index'] = extra_image_frame_index
            # print(
            #     f"return data with video shape: {data_dict['images'].shape}")
            return data_dict
        except Exception as e:
            print(
                f"Error loading scene {idx} from {meta_path}: {e}, loading next scene")
            return self.__getitem__((idx + 1) % len(self.meta_paths))
            # return None

    def __len__(self):
        return len(self.meta_paths)

    def load_image_into_memory(self, parser, indices, trajectories):
        # print(f"Loading {len(indices)} images into memory...")
        # print(f"Trajectories: {len(trajectories)}")

        data_list = []
        begin_index = indices[0]
        for i, index in enumerate(indices):
            image_path = parser.image_paths[index]
            image_name = parser.image_names[index]
            image = imageio.imread(image_path)[..., :3]
            height, width = image.shape[:2]
            camera_id = parser.camera_ids[index]
            K = parser.Ks_dict[camera_id].copy()
            params = parser.params_dict[camera_id]
            c2w = parser.camtoworlds[index]
            mask = parser.mask_dict[camera_id]
            w2c = np.linalg.inv(c2w)

            if len(params) > 0:
                mapx, mapy = parser.mapx_dict[camera_id], parser.mapy_dict[camera_id]
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w_roi, h_roi = parser.roi_undist_dict[camera_id]
                image = image[y: y + h_roi, x: x + w_roi]

            if len(self.patch_size) > 0:
                if len(trajectories[0]) == 4:
                    x_start, y_start, _, _ = trajectories[i]
                else:
                    x_start, y_start = trajectories[i]
                image = image[
                    y_start: y_start + int(self.patch_size[1]),
                    x_start: x_start + int(self.patch_size[0]),
                ]
                K[0, 2] -= x_start
                K[1, 2] -= y_start

            h, w = image.shape[:2]
            cam_info = CameraInfo(
                uid=i,
                colmapid=index,
                K=K,
                w2c=w2c,
                image_name=image_name,
                image_path=image_path,
                width=w,
                height=h,
            )
            # print(
            #     f"K,w2c for image {K.shape, w2c.shape} for image {image_name}")

            data = {
                "cam_info": cam_info,
                "image": torch.from_numpy(image) / 255.0,
                "image_name": image_name,
            }
            if mask is not None:
                data["mask"] = torch.from_numpy(mask)
            data_list.append(data)

        return data_list

    def generate_fixed_step_trajectory(self, parser, indices):

        num_images = len(indices)
        image_0 = cv2.imread(parser.image_paths[0])

        h, w = image_0.shape[:2]
        if h < self.patch_size[1] or w < self.patch_size[0]:
            raise ValueError(
                f"Image size ({h}, {w}) is smaller than patch size {self.patch_size}. Please adjust the patch size or use larger images.")
        # print(f"Image shape: ({h}, {w}), patch size: {self.patch_size}")
        # pdb.set_trace()
        # Test split: center crop
        if self.split == 'test':
            center_x = (w - self.patch_size[0]) // 2
            center_y = (h - self.patch_size[1]) // 2
            trajectories = [(center_x, center_y)] * num_images
            return trajectories

        original_state = np.random.get_state()
        np.random.seed()
        max_height = h - self.patch_size[1]
        max_width = w - self.patch_size[0]

        # print(
        #     f"Generating fixed step trajectory for {num_images} images with size ({w}, {h})")

        trajectories = []
        x = np.random.randint(0, max(w - int(self.patch_size[0]), 1))
        y = np.random.randint(0, max(h - int(self.patch_size[1]), 1))
        direction_x = 1
        direction_y = 1
        step_y = np.random.randint(1, 20)
        step_x = np.random.randint(5, 30)
        max_try = 500
        _try = 0
        while len(trajectories) < num_images:
            _try += 1
            if _try > max_try:
                print(
                    f"Max try reached: {_try}, stopping trajectory generation.")
                # Copy the res of test
                center_x = (w - self.patch_size[0]) // 2
                center_y = (h - self.patch_size[1]) // 2
                trajectories = [(center_x, center_y)] * num_images
                return trajectories
            # print(
            #     f"Current position: ({x}, {y}), direction: ({direction_x}, {direction_y}), step: ({step_x}, {step_y})")
            new_x = x + direction_x * step_x
            new_y = y + direction_y * step_y
            if new_x >= max_width or new_y >= max_height or new_x < 0 or new_y < 0:
                if new_x >= max_width and new_y >= max_height:
                    direction_x = -1
                    direction_y = -1
                elif new_x < 0 and new_y >= max_height:
                    direction_x = 1
                    direction_y = -1
                elif new_x >= max_width and new_y < 0:
                    direction_x = -1
                    direction_y = 1
                elif new_x < 0 and new_y < 0:
                    direction_x = 1
                    direction_y = 1
                elif new_y >= max_height:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = -1
                elif new_x >= max_width:
                    direction_x = -1
                    direction_y = np.random.choice([-1, 1])
                elif new_y < 0:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = 1
                elif new_x < 0:
                    direction_x = 1
                    direction_y = np.random.choice([-1, 1])

                step_y = np.random.randint(5, 30)
                step_x = np.random.randint(5, 30)
                continue
            else:
                x, y = new_x, new_y
            trajectories.append((x, y))
            # print(f"Generated trajectory point: ({x}, {y})")

        np.random.set_state(original_state)
        return trajectories[:num_images]

    # def visualize_point_projection_video(self, scene_idx: int, fixed_point_world: np.ndarray, save_root='video'):

    #     import cv2
    #     import numpy as np
    #     from tqdm import tqdm
    #     import os

    #     assert fixed_point_world.shape == (
    #         3,), "fixed_point_world must be a 3D point."

    #     meta_path = self.meta_paths[scene_idx]
    #     parsing_file_path = os.path.dirname(meta_path)
    #     parser = Parser(parsing_file_path, factor=1, normalize=True)

    #     # 构造连续帧的 index + trajectory
    #     total_frames = len(parser.image_names)
    #     possible_lengths = [21, 29, 41, 49, 61, 69, 81]
    #     sample_steps = [
    #         1,
    #         2,
    #         3,
    #         5,
    #         6,
    #         7,
    #     ]

    #     if self.split == 'train':
    #         required_frames = 0x3f3f3f
    #         chosen_length = np.random.choice(possible_lengths)
    #         while required_frames > total_frames:
    #             sample_step = np.random.choice(sample_steps)
    #             required_frames = (chosen_length - 1) * sample_step + 1
    #             # print(
    #             #     f"Required frames: {required_frames}, Total frames: {total_frames}, Sample step: {sample_step}")
    #         start = np.random.randint(0, total_frames - required_frames + 1)
    #         indices = np.arange(start, start + required_frames, sample_step)
    #     else:
    #         # 验证阶段默认固定长度和步长
    #         sample_step = 1  # 或者设置为其他默认值
    #         indices = np.arange(0, (41 - 1) * sample_step + 1, sample_step)

    #     print(
    #         f"Generating trajectory for scene {scene_idx} with indices: {indices}")
    #     trajectory = self.generate_fixed_step_trajectory(parser, indices)
    #     data_list = self.load_image_into_memory(parser, indices, trajectory)

    #     h, w = self.patch_size[1], self.patch_size[0]
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     fps = 10
    #     output_path = os.path.join(
    #         save_root, os.path.basename(os.path.dirname(meta_path))
    #     )

    #     save_dir = os.path.dirname(output_path)
    #     os.makedirs(save_dir, exist_ok=True)

    #     video_writer = cv2.VideoWriter(output_path+'.mp4', fourcc, fps, (w, h))

    #     for data in tqdm(data_list, desc="Visualizing projection video"):
    #         img = (data["image"].numpy() * 255).astype(np.uint8)
    #         if img.shape[0] == 3:  # C,H,W
    #             img = np.transpose(img, (1, 2, 0))
    #         img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #         cam_info = data["cam_info"]
    #         K = cam_info.K
    #         w2c = cam_info.w2c

    #         # 把固定点投影到像素坐标
    #         P_world_h = np.hstack([fixed_point_world, 1])
    #         P_cam_h = w2c @ P_world_h
    #         P_cam = P_cam_h[:3]

    #         if P_cam[2] <= 0:
    #             video_writer.write(img_bgr)
    #             continue

    #         p_img_h = K @ (P_cam / P_cam[2])
    #         u, v = int(p_img_h[0]), int(p_img_h[1])
    #         center_x, center_y = w // 2, h // 2

    #         u = np.clip(u, 0, w - 1)
    #         v = np.clip(v, 0, h - 1)

    #         cv2.arrowedLine(img_bgr, (center_x, center_y), (u, v), color=(
    #             0, 255, 0), thickness=2, tipLength=0.2)
    #         cv2.circle(img_bgr, (u, v), radius=5,
    #                    color=(0, 0, 255), thickness=-1)

    #         video_writer.write(img_bgr)

    #     video_writer.release()
    #     print(f"Projection video saved to {output_path}")


if __name__ == "__main__":
    split = 'train'
    dataset = ScenesDataset(split=split, ratio=1, patch_size=[832, 480])
    num_scene = len(dataset)
    for data in dataset:
        print(f"Frame number: {len(data['images'])}")
        print(f"Extra frame number: {len(data['extra_images'])}")
        # print(f"shape of input_images: {data['images'].shape}")
        # print(f"shape of extra_images: {data['extra_images'].shape}")
    # valid_json = []
    # invalid_json = []

    # print(f"Using {mp.cpu_count()} processes for loading {num_scene} scenes.")

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     result_iterator = pool.imap_unordered(process_scene, range(num_scene))
    #     with tqdm(total=num_scene) as pbar:
    #         results = []
    #         for res in result_iterator:
    #             results.append(res)
    #             pbar.update(1)

    # for r in results:
    #     res = r.get()
    #     if res[1]:
    #         valid_json.append(res[2])
    #     else:
    #         invalid_json.append(res[2])

    # with open('valid_scenes.json', 'w') as f:
    #     f.write('\n'.join(valid_json))
    # with open('invalid_scenes.json', 'w') as f:
    #     f.write('\n'.join(invalid_json))
