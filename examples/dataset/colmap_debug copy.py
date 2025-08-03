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


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        # print(f"Data dir{data_dir}, factor {factor}, normalize {normalize}, test_every {test_every}")
        colmap_dir = os.path.join(data_dir, "colmap/sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse/0")

        assert os.path.exists(colmap_dir), (
            f"COLMAP directory {colmap_dir} does not exist."
        )

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate(
                [np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array(
                    [cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array(
                    [cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert camtype == "perspective" or camtype == "fisheye", (
                f"Only perspective and fisheye cameras are supported, got {type_}"
            )

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (
                cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]
        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        # print("factor", factor)
        # print("self.extconf", self.extconf)
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        if os.path.exists(os.path.join(data_dir, "images")):
            colmap_image_dir = os.path.join(data_dir, "images")
        else:
            colmap_image_dir = os.path.join(data_dir, "images_4")
        # TODO
        image_dir = os.path.join(data_dir, "images_4" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f])
                       for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            # print(f"normalizing the world space with factor {factor}")
            T1 = similarity_from_cameras(camtoworlds)
            # print(f"Initial transform camtoworlds:\n{camtoworlds}")
            camtoworlds = transform_cameras(T1, camtoworlds)
            # print(f"Transformed camtoworlds:\n{camtoworlds}")
            points = transform_points(T1, points)

            # T2 = align_principle_axes(points)
            # camtoworlds = transform_cameras(T2, camtoworlds)
            # points = transform_points(T2, points)

            # transform = T2 @ T1
            transform = T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        # Dict[str, np.ndarray], image_name -> [M,]
        self.point_indices = point_indices
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert camera_id in self.params_dict, (
                f"Missing params for camera {camera_id}"
            )
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class ScenesDataset(Dataset):
    """A simple dataset class."""

    def __init__(
        self,
        no_extra_frame=False,
        max_frame=81,
        min_frame=21,
        relative_pose=True,
        split='train',
        ratio=0.99,
        patch_size: Optional[List[int]] = [720, 480],  # W H
    ):
        self.no_extra_frame = no_extra_frame
        self.max_frame = max_frame
        self.min_frame = min_frame
        print(f"Frame range: {self.min_frame} to {self.max_frame}")
        if self.no_extra_frame:
            print(
                "Info: no_extra_frame is set to True, extra_images and extra_image_frame_index will be None.")
        else:
            print(
                "Info: no_extra_frame is set to False, extra_images and extra_image_frame_index will be used.")
        self.split = split
        with open(meta_json, 'r') as f:
            meta_paths = f.readlines()
        self.meta_paths = [p.strip() for p in meta_paths]
        assert ratio <= 1.0, f"Ratio {ratio} should be less than or equal to 1.0"

        total_scenes = len(self.meta_paths)
        # shuffle
        # random.shuffle(self.meta_paths)
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
        # print(f"Loading scene {idx}...")
        scene_num = len(self.meta_paths)
        idx = (idx+scene_num) % scene_num
        meta_path = self.meta_paths[idx]
        try:
            # print(f"Loading scene {idx} from {meta_path}")
            parsing_file_path = os.path.dirname(meta_path)
            # print(f"Parsing file path: {parsing_file_path}")
            parser = Parser(parsing_file_path, factor=1, normalize=True)

            total_frames = len(parser.image_names)
            if total_frames < 21:
                raise ValueError(
                    f"Total frames {total_frames} is less than 21, cannot sample trajectory.")

            # possible_lengths = list(range(21, 81+1, 4))
            # possible_lengths = [13:65]
            possible_lengths = list(range(
                self.min_frame, self.max_frame + 1, 4))  # [21, 25, ..., 81]
            sample_steps = [
                1,
                2,
                3,
                4,
                5,
                6
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
                            0, (self.min_frame - 1) * sample_step + 1, sample_step)
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
                indices = np.arange(0, (self.min_frame - 1)
                                    * sample_step + 1, sample_step)

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

            if self.no_extra_frame:
                data_dict['extra_images'] = None
                data_dict['extra_image_frame_index'] = None
            elif self.split == 'train':
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
                if extra_image_frame_index.ndim == 0:
                    data_dict['extra_images'] = None
                    data_dict['extra_image_frame_index'] = None
                else:
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
                assert frame_deacivate > 2
                valid_mask[frame_deacivate-1::frame_deacivate] = True

                # Create the 'extra_images' and 'extra_image_frame_index' based on the valid mask
                # Select valid images based on the mask
                extra_images = data_dict['images'][valid_mask]
                extra_image_frame_index = torch.nonzero(
                    valid_mask, as_tuple=False).squeeze()  # Get indices of valid frames
                # print(
                #     f"Extra images shape: {extra_images.shape}, extra_image_frame_index shape: {extra_image_frame_index.shape}")
                # print(f"extra index ndim: {extra_image_frame_index.ndim}")
                if extra_image_frame_index.ndim == 0:
                    data_dict['extra_images'] = None
                    data_dict['extra_image_frame_index'] = None
                else:
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
            # print(
            #     f"Error loading scene {idx} from {meta_path}: {e}, loading next scene")
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
        step_y = np.random.randint(0, 6+1)
        step_x = np.random.randint(0, 9+1)
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

    def visualize_point_projection_video(self, scene_idx: int, fixed_point_world: np.ndarray, save_root='video'):

        import cv2
        import numpy as np
        from tqdm import tqdm
        import os

        assert fixed_point_world.shape == (
            3,), "fixed_point_world must be a 3D point."

        meta_path = self.meta_paths[scene_idx]
        parsing_file_path = os.path.dirname(meta_path)
        parser = Parser(parsing_file_path, factor=1, normalize=True)

        image_0 = cv2.imread(parser.image_paths[0])
        img_h, img_w = image_0.shape[:2]
        if img_h < self.patch_size[1] or img_w < self.patch_size[0]:
            raise ValueError(
                f"Image size ({img_h}, {img_w}) is smaller than patch size {self.patch_size}. Please adjust the patch size or use larger images.")
        # 构造连续帧的 index + trajectory
        total_frames = len(parser.image_names)
        possible_lengths = list(range(
            self.min_frame, self.max_frame + 1, 4))  # [21, 25, ..., 81]
        sample_steps = [
            1,
            2,
            3,
            5,
            6,
            7,
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
                        0, (self.min_frame - 1) * sample_step + 1, sample_step)
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
            indices = np.arange(0, (self.min_frame - 1)
                                * sample_step + 1, sample_step)

        print(
            f"Generating trajectory for scene {scene_idx} with indices: {indices}")

        trajectory = self.generate_fixed_step_trajectory(parser, indices)
        data_list = self.load_image_into_memory(parser, indices, trajectory)

        h, w = self.patch_size[1], self.patch_size[0]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 10
        output_path = os.path.join(
            save_root, os.path.basename(os.path.dirname(meta_path))
        )

        save_dir = os.path.dirname(output_path)
        os.makedirs(save_dir, exist_ok=True)

        video_writer = cv2.VideoWriter(output_path+'.mp4', fourcc, fps, (w, h))

        for data in tqdm(data_list, desc="Visualizing projection video"):
            img = (data["image"].numpy() * 255).astype(np.uint8)
            if img.shape[0] == 3:  # C,H,W
                img = np.transpose(img, (1, 2, 0))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cam_info = data["cam_info"]
            K = cam_info.K
            w2c = cam_info.w2c

            # 把固定点投影到像素坐标
            P_world_h = np.hstack([fixed_point_world, 1])
            P_cam_h = w2c @ P_world_h
            P_cam = P_cam_h[:3]

            if P_cam[2] <= 0:
                video_writer.write(img_bgr)
                continue

            p_img_h = K @ (P_cam / P_cam[2])
            u, v = int(p_img_h[0]), int(p_img_h[1])
            center_x, center_y = w // 2, h // 2

            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)

            cv2.arrowedLine(img_bgr, (center_x, center_y), (u, v), color=(
                0, 255, 0), thickness=2, tipLength=0.2)
            cv2.circle(img_bgr, (u, v), radius=5,
                       color=(0, 0, 255), thickness=-1)

            video_writer.write(img_bgr)

        video_writer.release()
        print(f"Projection video saved to {output_path}")


if __name__ == "__main__":
    split = 'test'
    dataset = ScenesDataset(split=split, ratio=0.01, patch_size=[720, 480])

    num_scene = len(dataset)
    print(f"Total {num_scene} scenes in the dataset for {split} split.")
    for idx, data in enumerate(dataset):
        print(f"Processing scene {idx} with data: {data['images'].shape}")

#  python -m examples.dataset.colmap_debug
