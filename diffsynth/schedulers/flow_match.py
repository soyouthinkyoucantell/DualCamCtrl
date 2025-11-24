import torch


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None,methods='naive'):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if methods=='naive':
            self.sigmas = self.shift * self.sigmas / \
                (1 + (self.shift - 1) * self.sigmas)
        elif methods in ('early', 'mid', 'late'):
            # 绝对 timestep 轴上做分段线性： [t_lo, 750), [750, 900), [900, t_hi]
            # 并按 4:1:1 分配 step 数，严格保证计数
            total = num_inference_steps
            device = self.sigmas.device if hasattr(self, "sigmas") else None

            # 顶部与底部边界（受 denoising_strength 影响的上界）
            t_lo = float(self.sigma_min * self.num_train_timesteps)  # ~2.994
            t_hi = float( (self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength)
                          * self.num_train_timesteps )               # <= 1000

            # 三段的绝对边界（中段、末段固定为 750/900；首段下界为 t_lo，末段上界为 t_hi）
            b0_lo, b0_hi = t_lo, 750.0
            b1_lo, b1_hi = 750.0, 900.0
            b2_lo, b2_hi = 900.0, t_hi

            # 防止上界<下界（例如 t_hi<900 的极端情况）
            def _clamp_segment(lo, hi):
                lo2 = max(lo, t_lo)
                hi2 = min(hi, t_hi)
                if hi2 < lo2:  # 该段被“挤没了”
                    hi2 = lo2
                return lo2, hi2

            b0_lo, b0_hi = _clamp_segment(b0_lo, b0_hi)
            b1_lo, b1_hi = _clamp_segment(b1_lo, b1_hi)
            b2_lo, b2_hi = _clamp_segment(b2_lo, b2_hi)

            # 4:1:1 份额
            w_early, w_mid, w_late = 1, 1, 1
            if methods == 'early':
                w_early = 4
            elif methods == 'mid':
                w_mid = 4
            else:  # 'late'
                w_late = 4

            w_sum = w_early + w_mid + w_late  # = 6
            n0 = int(round(total * (w_early / w_sum)))
            n1 = int(round(total * (w_mid   / w_sum)))
            n2 = total - n0 - n1  # 收尾，确保总和==total

            # 为了满足采样单调从噪声到净化（通常从高->低），我们让每段都“高到低”取点
            # 注意避免相邻段的端点重复：后续段 drop_first=True
            def seg_desc(lo, hi, n, drop_first):
                if n <= 0:
                    return torch.empty(0, device=device)
                # 该段的上界是 hi，下界是 lo；我们从 hi -> lo 线性取 n 个点
                seg = torch.linspace(hi, lo, n, device=device)
                return seg[1:] if drop_first and seg.numel() > 0 else seg

            # 生成三段 timesteps（单位：绝对 0..1000 轴）
            t2 = seg_desc(b2_lo, b2_hi, n2, drop_first=False)                   # [900, t_hi]
            t1 = seg_desc(b1_lo, b1_hi, n1, drop_first=(t2.numel() > 0))        # [750, 900)
            t0 = seg_desc(b0_lo, b0_hi, n0, drop_first=(t2.numel()+t1.numel()>0))  # [t_lo, 750)

            t_all = torch.cat([t2, t1, t0], dim=0)  # 从高到低拼接，避免回跳

            # 若因 round 出现 ±1 的长度偏差，做一次截断/补齐
            if t_all.numel() > total:
                t_all = t_all[:total]
            elif t_all.numel() < total:
                if t_all.numel() == 0:
                    t_all = torch.full((total,), t_hi, device=device)
                else:
                    t_all = torch.cat([t_all, t_all.new_tensor([t_all[-1]] * (total - t_all.numel()))], dim=0)

            # timesteps -> sigmas
            self.timesteps = t_all
            self.sigmas = self.timesteps / self.num_train_timesteps
    
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        # print(f"Sample shape: {original_samples.shape}, Noise shape: {noise.shape}, Sigma: {sigma}")
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import os 

    def _noise_levels_from_sigmas(sigmas: torch.Tensor):
        s = sigmas.detach().cpu().numpy()
        s_hi, s_lo = float(s.max()), float(s.min())
        if np.isclose(s_hi, s_lo):
            return np.zeros_like(s)
        # 归一化到 [0,1]，1=顶部（噪声大），0=底部
        nl = (s - s_lo) / (s_hi - s_lo)
        return nl

    def _gaussian_kernel1d(sigma_bins: float):
        k = int(max(3, 6 * sigma_bins)) | 1  # 覆盖 ±3σ，且为奇数
        half = k // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        ker = np.exp(-0.5 * (x / sigma_bins) ** 2)
        ker /= ker.sum()
        return ker

    def _adaptive_kde_density(samples_01: np.ndarray, height: int, base_bw_bins=1.5, bw_alpha=1.2):
        """
        使用自适应带宽的一维 KDE。
        - samples_01: 已归一化到 [0,1] 的样本位置（越靠近 1 越接近顶部）
        - height: 输出直方图（竖直分辨率）
        - base_bw_bins: 基础带宽（单位=bin）；越大越平滑
        - bw_alpha: 带宽放大系数，乘在最近邻间距上（自适应）
        返回：长度为 height 的密度，并已归一化到 [0,1]
        """
        samples = np.clip(samples_01, 0.0, 1.0)
        samples.sort()

        # 计算每个样本的最近邻间距（在 [0,1] 空间）
        if len(samples) > 1:
            left_gap  = np.diff(np.concatenate([[samples[0]], samples]))
            right_gap = np.diff(np.concatenate([samples, [samples[-1]]]))
            nn_gap = np.minimum(left_gap[1:], right_gap[:-1])
            # 边界两个点
            nn_gap = np.concatenate([[right_gap[0]], nn_gap, [left_gap[-1]]])
        else:
            nn_gap = np.array([1.0])

        # 将最近邻间距换算到“bin”为单位的带宽，并与基础带宽取最大，避免过窄
        # 注意：一个 bin 的“物理长度”= 1/height
        min_bw_bins = base_bw_bins
        adaptive_bw_bins = np.maximum(min_bw_bins, bw_alpha * nn_gap * height)

        # 评估位置（bin 中心）
        centers = (np.arange(height) + 0.5) / height

        # KDE：对每个样本贡献一个高斯核，带宽使用自适应带宽
        dens = np.zeros_like(centers, dtype=np.float64)
        inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)
        for xi, bw_bins in zip(samples, adaptive_bw_bins):
            # 把带宽从“bin”单位转回到 [0,1] 空间
            bw = bw_bins / height
            # 避免 0 带宽
            bw = max(bw, 1e-6)
            z = (centers - xi) / bw
            dens += inv_sqrt2pi * np.exp(-0.5 * z * z) / bw

        # 归一化到 [0,1] 以便做热度图
        if dens.max() > 0:
            dens = dens / dens.max()
        return dens

    def _build_image_from_density(density_0to1: np.ndarray, width: int, aa_sigma_bins=0.8):
        """
        将竖直方向的密度向量扩展为宽度为 width 的图像，并做一次小带宽高斯卷积用于抗锯齿。
        """
        # 抗锯齿：对密度做一次轻微高斯卷积（按“bin”为单位）
        ker = _gaussian_kernel1d(aa_sigma_bins)
        smoothed = np.convolve(density_0to1, ker, mode="same")

        # 生成二维图像，并翻转使得 1 在图像顶部
        img = np.tile(smoothed.reshape(-1, 1), (1, width))
        img = np.flipud(img)
        return img

    scheduler = FlowMatchScheduler(
        num_inference_steps=50,
        num_train_timesteps=1000,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    )
    methods = ['early', 'mid', 'late']

    # 可调可玩的参数 -----------------------------
    HEIGHT = 800     # 竖直分辨率（bin 数），越大越细腻
    WIDTH  = 80      # 横向“加粗”宽度
    OVERSAMPLE = 20  # 每对相邻样本之间插入的虚拟样本数（超采样）
    BASE_BW_BINS = 2.5  # 基础带宽（单位=bin），越大越平滑
    BW_ALPHA = 1.8      # 自适应带宽系数；越大在稀疏处越宽
    AA_SIGMA = 0.9      # 最终抗锯齿的高斯 σ（单位=bin）
    # -------------------------------------------

    for method in methods:
        scheduler.set_timesteps(num_inference_steps=48, methods=method)
        print(f"Method: {method}, Timesteps: {scheduler.timesteps}")

        # ---- 统计各区间数量（基于 training timesteps）----
        ts = scheduler.timesteps.detach().cpu().numpy()
        bins = [(0, 750), (750, 900), (900, 1000)]  # 左闭右开，最后一个区间右端包含
        counts = []
        for (lo, hi) in bins:
            if hi == 1000:
                mask = (ts >= lo) & (ts <= hi)
            else:
                mask = (ts >= lo) & (ts < hi)
            counts.append(int(mask.sum()))

        total_steps = len(ts)
        early_cnt, mid_cnt, late_cnt = counts
        assert early_cnt + mid_cnt + late_cnt == total_steps, "区间计数与总步数不一致！"

        # 打印统计信息
        pct = [c / total_steps * 100.0 for c in counts]
        print(
            f"Counts by intervals [0,750), [750,900), [900,1000]: "
            f"{counts}  |  percents: {[f'{p:.1f}%' for p in pct]}"
        )

        # 保存成 CSV
        csv_path = os.path.join('output/scheduler', f"{method}_interval_counts.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("interval,count,percent\n")
            names = ["0-750", "750-900", "900-1000"]
            for name, c, p in zip(names, counts, pct):
                f.write(f"{name},{c},{p:.4f}\n")
            f.write(f"total,{total_steps},100.0000\n")
        print(f"Saved interval stats: {csv_path}")


