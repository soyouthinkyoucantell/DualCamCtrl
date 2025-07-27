#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
from .cameras import Camera
import numpy as np
from .general_utils import PILtoTorch
from .graphics_utils import fov2focal
from typing import NamedTuple
import cv2
from .graphics_utils import (
    focal2fov,
    getWorld2View2,
    getProjectionMatrixShift,
    getProjectionMatrix,
)

WARNED = False


class Pose:
    def __init__(self, K: np.ndarray, w2c: np.ndarray, image_name: str = "", image_path: str = "", width: int = 0, height: int = 0):
        assert K.shape == (3, 3), "K must be a 3x3 matrix"
        assert w2c.shape == (4, 4), "w2c must be a 4x4 matrix"

        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        self.w2c_mat = w2c.copy()
        self.c2w_mat = np.linalg.inv(self.w2c_mat)

        self.image_name = image_name
        self.image_path = image_path
        self.width = width
        self.height = height


class CameraInfo(NamedTuple):
    uid: int
    colmapid: int
    K: np.array
    w2c: np.array
    image_name: str
    image_path: str
    width: int
    height: int


class SampleCamera(nn.Module):
    def __init__(
        self, cam_info, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
    ):
        super(SampleCamera, self).__init__()
        w2c = cam_info.w2c
        self.R = w2c[:3, :3].T
        self.T = w2c[:3, 3]
        self.image_width = cam_info.width  # .item()
        self.image_height = cam_info.height  # .item()

        K = cam_info.K
        cx_cy = K[[0, 1], [2, 2]]

        self.FoVx = focal2fov(K[0, 0], self.image_width)
        self.FoVy = focal2fov(K[1, 1], self.image_height)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T, trans, scale))
            .transpose(0, 1)
            .cuda()
        )
        self.projection_matrix = (
            getProjectionMatrixShift(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
                width=self.image_width,
                height=self.image_height,
                cxcy=cx_cy,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera(nn.Module):
    def __init__(
        self, cam_info, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
    ):
        super(Camera, self).__init__()

        self.uid = cam_info.uid.item()
        self.colmap_id = cam_info.colmapid.item()

        w2c = cam_info.w2c[0]
        self.R = w2c[:3, :3].numpy().T
        self.T = w2c[:3, 3].numpy()
        self.image_name = cam_info.image_name[0]
        self.image_width = cam_info.width.item()
        self.image_height = cam_info.height.item()

        K = cam_info.K[0]
        cx_cy = K[[0, 1], [2, 2]]

        self.FoVx = focal2fov(K[0, 0], self.image_width)
        self.FoVy = focal2fov(K[1, 1], self.image_height)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T, trans, scale))
            .transpose(0, 1)
            .cuda()
        )
        self.projection_matrix = (
            getProjectionMatrixShift(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
                width=self.image_width,
                height=self.image_height,
                cxcy=cx_cy,
            )
            .transpose(0, 1)
            .cuda()
        )
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


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
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(
        B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker
