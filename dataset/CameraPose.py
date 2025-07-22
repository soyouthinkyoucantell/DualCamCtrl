import imageio

import open3d.visualization.rendering as rendering

import open3d as o3d
import numpy as np
import json
import os
from PIL import Image
os.environ["OPEN3D_RENDERING_HEADLESS"] = "true"


class ImagePoseDatasetDict:
    def __init__(self, json_paths, transform=None):
        """
        json_paths: 所有 JSON 路径的列表
        transform: torchvision.transforms
        """
        self.transform = transform

        self.meta_json_paths = []
        for json_path in json_paths:
            self.meta_json_paths.append(json_path)
        print(f"self.meta_json_paths: {self.meta_json_paths[:10]}")

    def __getitem__(self, idx):
        total_len = len(self.meta_json_paths)
        idx = (idx+total_len) % total_len
        return SceneImagePoseDataset(self.meta_json_paths[idx], transform=self.transform)

    def __len__(self):
        return len(self.meta_json_paths)


class SceneImagePoseDataset:
    def __init__(self, json_path, transform=None):
        self.json_path = json_path
        self.transform = transform
        self.base_dir = os.path.dirname(json_path)
        self.camera_data = CameraDataset(json_path)

        # 替换为完整路径
        for frame in self.camera_data.frames:
            frame.file_path = os.path.join(
                self.base_dir, frame.file_path.replace('images', 'images_4'))

    def __len__(self):
        return len(self.camera_data)

    def __getitem__(self, idx):
        frame = self.camera_data[idx]
        image = Image.open(frame.file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'pose': frame.pose,
            'path': frame.file_path,
            'camera_position': frame.camera_position(),
            'forward_vector': frame.forward_vector()
        }

    def get_all_paths(self):
        return [frame.file_path for frame in self.camera_data.frames]


class CameraDataset:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 相机内参
        self.w = data['w']
        self.h = data['h']
        self.fl_x = data['fl_x']
        self.fl_y = data['fl_y']
        self.cx = data['cx']
        self.cy = data['cy']
        self.k1 = data['k1']
        self.k2 = data['k2']
        self.p1 = data['p1']
        self.p2 = data['p2']
        self.camera_model = data.get('camera_model', 'OPENCV')

        # 所有帧
        self.frames = [CameraFrame(f) for f in data['frames']]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class CameraFrame:
    def __init__(self, frame_dict):
        self.file_path = frame_dict['file_path']
        self.colmap_id = frame_dict.get('colmap_im_id', None)
        self.transform_matrix = np.array(
            frame_dict['transform_matrix'], dtype=np.float32)  # 4x4

    @property
    def pose(self):
        """返回 4x4 的位姿矩阵"""
        return self.transform_matrix

    def rotation_matrix(self):
        return self.transform_matrix[:3, :3]

    def translation_vector(self):
        return self.transform_matrix[:3, 3]

    def camera_position(self):
        """返回相机在世界坐标系中的位置"""
        R = self.rotation_matrix()
        t = self.translation_vector()
        return -R.T @ t  # 世界坐标下相机中心

    def forward_vector(self):
        """相机朝向（z轴）"""
        return self.rotation_matrix()[:, 2]

    def up_vector(self):
        return self.rotation_matrix()[:, 1]

    def right_vector(self):
        return self.rotation_matrix()[:, 0]


def draw_camera_poses(c2ws, scale=0.1, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, c2w in enumerate(c2ws):
        origin = c2w[:3, 3]
        z_axis = c2w[:3, 2] * scale
        ax.quiver(*origin, *z_axis, color='b')  # camera forward
        ax.scatter(*origin, color='r', s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_image_plane(pose, image_path, scale=0.2):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 120))  # 缩小以更适配场景

    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    # 定义图像面片的局部平面网格（Z=0）
    x = np.linspace(-scale, scale, w)
    y = np.linspace(-scale * h / w, scale * h / w, h)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    colors = img_np.reshape(-1, 3)

    # 世界坐标变换
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points.T).T + t[None, :]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_world)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def save_rotating_scene_video(scene_dataset, output_video_path, downsample=5, image_scale=0.3,
                              width=1920, height=1080, num_views=60, fps=20):
    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    # 背景与光照
    scene.set_background([1.0, 1.0, 1.0, 1.0])  # 白背景
    scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 75000)
    scene.scene.enable_sun_light(True)

    # 从数据集中提取所有相机位置与朝向
    camera_positions = []
    camera_forwards = []

    for i in range(0, len(scene_dataset), downsample):
        sample = scene_dataset[i]
        c2w = sample['pose']
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        forward = R[:, 2]   # z轴 = 摄像头朝向
        camera_positions.append(t)
        camera_forwards.append(forward)

    camera_positions = np.stack(camera_positions)
    camera_forwards = np.stack(camera_forwards)

    # 获取一个“中心观察点”和整体观察方向
    avg_pos = camera_positions.mean(axis=0)
    avg_forward = camera_forwards.mean(axis=0)
    avg_target = avg_pos + avg_forward  # 相机朝向的前方点

    # 设置视角绕“camera front”转（从前方看相机）
    radius = 2.0  # 或根据距离自动计算
    up = np.array([0, 1, 0], dtype=np.float32)
    # 设置视角环绕路径
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = bounds.get_extent()
    radius = np.linalg.norm(extent) * 0.3

    video_frames = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = np.cos(angle) * radius
        z = np.sin(angle) * radius
        eye = avg_target + np.array([x, 0.4 * radius, z])  # 从前上方绕相机看

        renderer.setup_camera(
            60.0,
            avg_pos.astype(np.float32),  # 相机中心点（我们看的目标）
            eye.astype(np.float32),      # 虚拟观察者位置
            up
        )

        img = renderer.render_to_image()
        img_np = np.asarray(img)
        video_frames.append(img_np)

    # 保存为 MP4 视频
    imageio.mimsave(output_video_path, video_frames, fps=fps)
    print(f"🎥 Saved rotating scene video to: {output_video_path}")


def get_all_camera_poses(dataset_dict):
    """
    从 ImagePoseDatasetDict 提取所有场景中的所有相机位姿
    返回值：np.ndarray (N, 4, 4)
    """
    all_poses = []
    for i in range(len(dataset_dict)):
        scene_dataset = dataset_dict[i]
        for j in range(len(scene_dataset)):
            sample = scene_dataset[j]
            all_poses.append(sample['pose'])
    return np.stack(all_poses)


if __name__ == "__main__":
    import os
    import natsort

    total_json = []
    meta_json_path = '/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/camera_data_paths.json'

    if not os.path.exists(meta_json_path):
        data_root = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/DL3DV-10K/DL3DV-ALL-960P'
        print(
            f"Meta JSON file not found: {meta_json_path}, scanning directory: {data_root}")
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    total_json.append(json_path)
        total_json = natsort.natsorted(total_json)
        with open(meta_json_path, 'w') as f:
            for json_path in total_json:
                f.write(json_path + '\n')
    else:
        print(f"Meta JSON file found: {meta_json_path}, loading paths...")
        with open(meta_json_path, 'r') as f:
            total_json = [line.strip() for line in f.readlines()]

    print(f"len(total_json): {len(total_json)}")
    dataset = ImagePoseDatasetDict(total_json)

    # ✅ 每个场景逐个显示相机位姿
    output_dir = "camera_pose_viz"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(dataset)):
        scene_dataset = dataset[i]
        scene_name = os.path.basename(
            scene_dataset.json_path).replace(".json", "")
        video_path = os.path.join(output_dir, f"{i:03d}_{scene_name}.mp4")
        save_rotating_scene_video(scene_dataset, video_path, downsample=10)
