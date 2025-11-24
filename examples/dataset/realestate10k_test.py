from typing import Union
import cv2
import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from torch.utils.data.dataset import Dataset
from packaging import version as pver
from .colmap_debug import ray_condition, custom_meshgrid

# data_root = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/re10k/'
# train_dir = 'train'
# test_dir = 'test'
# train_root = os.path.join(data_root, train_dir)
# test_root = os.path.join(data_root, test_dir)
# re10k_train_meta_json = '/data/user/hongfeizhang/dataset/re10k/re10k_train_meta.json'
# re10k_test_meta_json = '/data/user/hongfeizhang/dataset/re10k/re10k_test_meta.json'


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class RealEstate10KPose(Dataset):
    def __init__(
        self,
        start=0,
        split="train",
        sample_stride=8,
        minimum_sample_stride=1,
        sample_n_frames=21,
        return_depth=True,
        relative_pose=True,
        sample_size=[320, 480],
        rescale_fxy=True,
        use_flip=False,
        no_extra_frame=True,
        use_image_depth=True,
        debug=False,
    ):
        self.use_image_depth = use_image_depth
        self.split = split
        self.return_depth = return_depth
        if self.split == "train":
            self.meta_json_path = (
                "/data/user/hongfeizhang/dataset/re10k/train_meta.json"
            )
            self.data_root = "/data/user/hongfeizhang/dataset/re10k/train_scenes"
        else:
            self.meta_json_path = "/data/user/hongfeizhang/dataset/re10k/test_meta.json"
            self.data_root = "/data/user/hongfeizhang/dataset/re10k/test_scenes"
        self.prompt_root = self.data_root.replace(
            f"{self.split}_scenes", f"{self.split}_captions"
        )
        self.no_extra_frame = no_extra_frame
        with open(self.meta_json_path, "r") as f:
            self.dataset = json.load(f)
        if debug:
            import random

            random.shuffle(self.dataset)
        print(f"Loaded {len(self.dataset)} samples from {self.meta_json_path}")
        self.dataset = self.dataset[start:]
        self.relative_pose = relative_pose
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames

        self.length = len(self.dataset)

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.sample_size = sample_size

        if use_flip:
            from torchvision.transforms import InterpolationMode

            pixel_transforms = [
                transforms.Resize(sample_size),
                RandomHorizontalFlipWithPose(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
            depth_transforms = [
                transforms.Resize(sample_size, interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlipWithPose(),
            ]

        else:
            pixel_transforms = [
                transforms.Resize(sample_size),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
            depth_transforms = [
                transforms.Resize(
                    sample_size, interpolation=transforms.InterpolationMode.NEAREST
                )
            ]

        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.depth_transforms = depth_transforms
        self.use_flip = use_flip

    def read_prompt(self, scene_name, end_frame_ind):
        prompt_dir = os.path.join(self.prompt_root, f"{scene_name}")
        with open(os.path.join(prompt_dir, "captions.json"), "r") as f:
            prompt_data = json.load(f)
        prompt_frame_ind = end_frame_ind // 60 * 60
        # print(
        #     f"Get prompt from frame index: {prompt_frame_ind} for end_frame_ind: {end_frame_ind}"
        # )
        if str(prompt_frame_ind) in prompt_data:
            return prompt_data[str(prompt_frame_ind)]

    def get_relative_pose(self, cam_params):
        # Always zero_init the first camera pose
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def decode_image(self, image_tensor):
        byte_data = image_tensor.numpy().tobytes()
        np_data = np.frombuffer(byte_data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return img

    def read_video(self, video_path):
        cap = cv2.VideoCapture(f"{video_path}")
        # print(f"Reading depth from video file: {video_path}")
        # 检查视频是否成功打开
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
            return

        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

        video_array = np.array(frames)
        cap.release()
        return video_array

    def get_batch(self, idx):
        #
        #   {
        #     "path": "01042371ee0b76ac.torch",
        #     "frame": 83,
        #     "height": 360,
        #     "width": 640,
        #     "key": "01042371ee0b76ac",
        #     "url": "https://www.youtube.com/watch?v=LiejcFr3v7c"
        #   },
        data_root = self.data_root
        current_sample_stride = self.sample_stride
        json_entry = self.dataset[idx]
        # print(
        #     f"Loading sample {idx}: {json_entry['key']} with {json_entry['frame']} frames")

        # Frame part
        total_frames = json_entry["frame"]
        # print(f"Total frames available: {total_frames}")
        if total_frames < self.sample_n_frames * self.minimum_sample_stride:
            raise ValueError(
                f"Total frames {total_frames} is less than sample_n_frames {self.sample_n_frames}"
            )

        if total_frames < self.sample_n_frames * current_sample_stride:
            if self.split == "test":
                raise ValueError(
                    f"Total frames {total_frames} is less than required {self.sample_n_frames* current_sample_stride}"
                )
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            if self.split == "train":
                current_sample_stride = random.randint(
                    self.minimum_sample_stride, maximum_sample_stride
                )
            else:
                current_sample_stride = maximum_sample_stride

        cropped_length = self.sample_n_frames * current_sample_stride
        if self.split == "train":
            # Randomly select a start frame index
            start_frame_ind = random.randint(
                0, max(0, total_frames - cropped_length - 1)
            )
        else:
            start_frame_ind = 0

        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        # frame_indices = np.linspace(
        #     start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int
        # )
        frame_indices = np.arange(
            start_frame_ind,
            end_frame_ind,
            (end_frame_ind - start_frame_ind) / self.sample_n_frames,
            dtype=int,
        )

        print(f"frame indices {frame_indices}")
        # Image and camera
        data_path = os.path.join(data_root, json_entry["path"])
        data = torch.load(data_path)

        cameras_info = data["cameras"]

        # Frame part
        scene_name = json_entry["key"]
        prompt = self.read_prompt(scene_name, end_frame_ind)

        # Camera
        cameras_info = torch.concat(
            [torch.zeros(cameras_info.shape[0], 1), cameras_info], dim=1
        )
        cam_params = [Camera(cameras_info[indice]) for indice in frame_indices]

        # image part
        images = data["images"]
        pixel_values = [self.decode_image(image_tensor) for image_tensor in images]
        pixel_values = np.stack(pixel_values, axis=0)  # [F, H, W, C]
        pixel_values = (
            torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        )  # [F, C, H, W]
        assert pixel_values.shape[0] == total_frames
        # print(f"pixel frames: {pixel_values.shape[0]}")
        pixel_values = pixel_values / 255.0  # Normalize to [0, 1]
        pixel_values = pixel_values[frame_indices]

        # depth part
        if self.return_depth:
            if self.use_image_depth:
                depth_dir = data_path.replace(
                    rf"{self.split}_scenes", rf"{self.split}_depth_maps"
                ).replace(".torch", "")
                depth_files = [
                    f
                    for f in os.listdir(depth_dir)
                    if os.path.isfile(os.path.join(depth_dir, f)) and f.endswith(".png")
                ]

                def extract_number(filename):
                    name_without_ext = os.path.splitext(filename)[0]
                    # print(f"name_without_ext: {name_without_ext}, type :{type(name_without_ext)}")
                    name_without_ext_int = int(name_without_ext)
                    # print(f"name_without_ext: {name_without_ext_int}, type :{type(name_without_ext_int)}")

                    return name_without_ext_int

                depth_files_sorted = sorted(depth_files, key=extract_number)
                depth_files = [
                    os.path.join(depth_dir, _depth_file_path)
                    for _depth_file_path in depth_files_sorted
                ]
                depth_numpy = [cv2.imread(_depth_path) for _depth_path in depth_files]
            else:
                depth_dir = data_path.replace(
                    rf"{self.split}_scenes", rf"{self.split}_video_depth_maps"
                ).replace(".torch", "")
                depth_video_file = os.path.join(depth_dir, "depth_vitl_fp16.mp4")
                depth_numpy = self.read_video(depth_video_file)
            # print(f"Depth frames: {len(depth_numpy)}")
            assert depth_numpy.shape[0] == total_frames
            depth_array = depth_numpy / 255.0
            depth_tensor = torch.from_numpy(depth_array).float().permute(0, 3, 1, 2)
            depth_tensor = depth_tensor[frame_indices]
        else:
            depth_tensor = torch.zeros_like(pixel_values)

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:  # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:  # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

        intrinsics = np.asarray(
            [
                [
                    cam_param.fx * self.sample_size[1],
                    cam_param.fy * self.sample_size[0],
                    cam_param.cx * self.sample_size[1],
                    cam_param.cy * self.sample_size[0],
                ]
                for cam_param in cam_params
            ],
            dtype=np.float32,
        )

        intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array(
                [cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32
            )
        # [1, n_frame, 4, 4]
        c2w = torch.as_tensor(c2w_poses)[None]
        c2w_tensor = torch.as_tensor(c2w_poses)
        intrinsics_tensor = intrinsics[0]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)

        else:
            flip_flag = torch.zeros(
                self.sample_n_frames, dtype=torch.bool, device=c2w.device
            )
        plucker_embedding = (
            ray_condition(
                intrinsics,
                c2w,
                self.sample_size[0],
                self.sample_size[1],
                device="cpu",
                flip_flag=flip_flag,
            )[0]
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return (
            pixel_values,
            depth_tensor,
            plucker_embedding,
            flip_flag,
            prompt,
            c2w_tensor,
            intrinsics_tensor,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        (
            video,
            depth_tensor,
            plucker_embedding,
            flip_flag,
            prompt,
            c2w_tensor,
            intrinsics_tensor,
        ) = (None, None, None, None, None, None, None)
        while True:
            try:
                idx = idx % self.length
                (
                    video,
                    depth_tensor,
                    plucker_embedding,
                    flip_flag,
                    prompt,
                    c2w_tensor,
                    intrinsics_tensor,
                ) = self.get_batch(idx)
                break
            except Exception as e:
                if self.split == "train":
                    idx = random.randint(0, self.length - 1)
                else:
                    idx += 32

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            depth_tensor = self.depth_transforms[0](depth_tensor)
            depth_tensor = self.depth_transforms[1](depth_tensor, flip_flag)

            # video = self.pixel_transforms[2](video)

        else:
            for transform in self.pixel_transforms:
                video = transform(video)
            for _depth_transform in self.depth_transforms:
                depth_tensor = _depth_transform(depth_tensor)

        sample = {
            "images": video,  # F C H W format
            "control": depth_tensor,  # F C H W format
            # Align dimensions
            "camera_infos": plucker_embedding.permute(1, 0, 2, 3),
            "prompt": prompt,
            "extrinsinc": c2w_tensor,
            "intrinsinc": intrinsics_tensor,
        }
        if self.no_extra_frame:
            sample["extra_images"] = None
            sample["extra_image_frame_index"] = None
        return sample


if __name__ == "__main__":
    # Example usage
    save_root = "video/realestate10k_pose"
    os.makedirs(save_root, exist_ok=True)
    dataset = RealEstate10KPose(
        split="test",
        start=0,
        sample_stride=1,
        return_depth=True,
        sample_n_frames=121,
        relative_pose=True,
        sample_size=[480, 720],
        rescale_fxy=True,
        use_flip=False,
        no_extra_frame=True,
        use_image_depth=False,
    )


    def custom_collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            values = [d[key] for d in batch]

            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                collated[key] = values
            elif values[0] is None:
                collated[key] = None
            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(values[0])}")
        return collated

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     collate_fn=custom_collate_fn,
    # )
    save_path = '/data/user/hongfeizhang/hongfei_workspace/DiffSynth-Studio/output/for_reconstruction/validation_results_50_inference_steps'
    for idx in range(len(dataset)):
        _intrinsic= dataset[idx]['intrinsinc']
        _extrinsinc = dataset[idx]['extrinsinc']
        print(f"{_intrinsic.shape}")
        print(f"{_extrinsinc.shape}")
        camera_data = {}

        # Store intrinsic parameters at the top level
        camera_data["intrinsic"] = _intrinsic.tolist()[0]  # Convert tensor to list for JSON compatibility

        # Store extrinsic parameters by frame number
        camera_data["extrinsic"] = {}

        for frame_idx in range(_extrinsinc.shape[0]):
            # Convert each extrinsic matrix (4x4) to a list and store it under the corresponding frame index
            camera_data["extrinsic"][f"{frame_idx}"] = _extrinsinc[frame_idx].tolist()

        # Define the save path for the JSON file
        _save_root = os.path.join(save_path, f"{idx}")
        os.makedirs(_save_root, exist_ok=True)
        json_save_path = os.path.join(_save_root, "gt_cam_info.json")

        # Save the camera data to a JSON file
        with open(json_save_path, "w") as json_file:
            json.dump(camera_data, json_file, indent=4)

        print(f"Camera info saved to {json_save_path}")
        images_save_root = os.path.join(_save_root,'gt_images')
        os.makedirs(images_save_root,exist_ok=True)
        from torchvision.utils import save_image

        for idx, _image in enumerate(dataset[idx]['images']):
            # Convert _image into a tensor if it's not already (optional depending on your dataset)
            if isinstance(_image, torch.Tensor):
                img_tensor = _image
            else:
                img_tensor = torch.tensor(_image)  # assuming _image is in a format convertible to tensor

            # Save the image with the naming convention "0.jpg", "1.jpg", etc.
            image_save_path = os.path.join(images_save_root, f"{idx}.png")
            save_image(img_tensor, image_save_path)
            print(f"Image {idx} saved to {image_save_path}")