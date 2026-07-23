import math

import torch.nn as nn
import torch


class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class CrossNet(nn.Module):
    """Deep & Cross Network (DCN) Cross layers for explicit feature interaction.

    Each cross layer applies: x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l
    This captures bounded-degree feature crosses efficiently without
    exponential parameter growth.
    """

    def __init__(self, input_dim, num_cross_layers=3):
        super(CrossNet, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.w = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, 1)) for _ in range(num_cross_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim)) for _ in range(num_cross_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for w, b in zip(self.w, self.b):
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

    def forward(self, x):
        x_0 = x
        for i in range(self.num_cross_layers):
            xw = torch.matmul(x, self.w[i])           # (batch, 1)
            x = x_0 * (xw + self.b[i].unsqueeze(0)) + x
        return x


class GazeMoE(nn.Module):
    """GazeMoE —— 门控混合专家视线模型（Mixture of Experts for Gaze）。

    架构（两个完全同构的专家）：
      - Iris 专家：全 124 维 → CrossNet + 4 ResBlocks → y_iris（虹膜偏转贡献量）
      - Head 专家：全 124 维 → CrossNet + 4 ResBlocks → y_head（头部姿态贡献量，base 预测）
      - Gate 路由：基于 ||rel|| 物理先验的二值化门控（STE）
        * prior_gate：sigmoid 锐化 + STE 二值化（0/1）
        * residual_gate：可学习微调（限幅，默认 0）
      - 输出 = y_head + gate * y_iris

    说明：两分支同构，不再对 head 做人工特征筛选，靠 gate 做路由分工。
    """

    # 用于门控权重的"虹膜相对眼眶中心偏移"特征索引
    # 左眼 rel_x=8, rel_y=9；右眼 rel_x=16, rel_y=17
    GATE_FEATURE_INDICES = (8, 9, 16, 17)

    # —— 门控参数（Otsu + EMA 自适应阈值）——
    OTSU_N_BINS   = 50
    EMA_MOMENTUM  = 0.99
    OTSU_MIN_SAMPLES = 16

    K_SHARP = 5.0
    GATE_RESIDUAL_SCALE = 0.0

    def __init__(self, input_size=124, output_size=2, dropout=0.3,
                 num_cross_layers=1, hidden_dim=128):
        super(GazeMoE, self).__init__()

        self.input_size = input_size

        # ============ 输入特征标准化参数（作为 buffer，随 state_dict 存取）============
        # 124 维特征量纲差异极大（face_area 上万、角度 ±90、ratio 0~1、gaze_vec ±1），
        # 直接喂给 Linear/CrossNet 会让大量纲特征主导梯度。这里对【专家分支】的输入
        # 做逐维标准化 (x-mean)/std；门控分支仍用原始特征（保留 Otsu 物理尺度）。
        # 默认 mean=0/std=1 → 恒等变换，兼容未 fit 的旧 checkpoint。
        self.register_buffer("feat_mean", torch.zeros(input_size))
        self.register_buffer("feat_std", torch.ones(input_size))
        self.register_buffer("feat_norm_ready", torch.tensor(0, dtype=torch.long))

        block_in = hidden_dim + input_size    # 128 + 124 = 252

        # ============ IRIS 专家 ============
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_net = CrossNet(input_size, num_cross_layers=1)

        self.block1 = ResidualBlock(block_in, hidden_dim, dropout)
        self.block2 = ResidualBlock(hidden_dim, 64, dropout)
        self.block3 = ResidualBlock(64, 32, dropout)
        self.block4 = ResidualBlock(32, 16, dropout)

        # 主干 16 维 + y_head 2 维（detach 拼接 anchor）→ y_iris
        self.output_layer = nn.Linear(16 + output_size, output_size)

        # ============ HEAD 专家（与 IRIS 完全同构）============
        # 同样看全 124 维，同样的 CrossNet + 4 ResBlocks 结构。
        # 输出层不拼 anchor（head 是 base，不需要 y_iris 作参考）。
        self.head_input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_cross_net = CrossNet(input_size, num_cross_layers=1)

        self.head_block1 = ResidualBlock(block_in, hidden_dim, dropout)
        self.head_block2 = ResidualBlock(hidden_dim, 64, dropout)
        self.head_block3 = ResidualBlock(64, 32, dropout)
        self.head_block4 = ResidualBlock(32, 16, dropout)
        self.head_output_layer = nn.Linear(16, output_size)

        # ============ Gate 层 ============
        gate_in = len(self.GATE_FEATURE_INDICES) * 2
        self.gate_residual = nn.Sequential(
            nn.Linear(gate_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

        # rel BatchNorm 去 DC
        self.rel_norm = nn.BatchNorm1d(len(self.GATE_FEATURE_INDICES), affine=False)

        # Otsu + EMA buffers
        self.register_buffer("threshold_ema",  torch.tensor(1.5))
        self.register_buffer("offset_ema_std", torch.tensor(0.5))

        # gate 索引 buffer
        self.register_buffer(
            "_gate_idx_buf",
            torch.tensor(self.GATE_FEATURE_INDICES, dtype=torch.long),
            persistent=False,
        )

    def set_feature_stats(self, mean, std):
        """写入输入标准化统计量（在训练集上拟合后调用）。

        mean/std: 形如 (input_size,) 的张量或可转张量的序列。
        std 会被 clamp 到一个下限，避免近常数维度放大噪声。
        """
        with torch.no_grad():
            mean_t = torch.as_tensor(mean, dtype=self.feat_mean.dtype,
                                     device=self.feat_mean.device).flatten()
            std_t = torch.as_tensor(std, dtype=self.feat_std.dtype,
                                    device=self.feat_std.device).flatten()
            std_t = torch.clamp(std_t, min=1e-6)
            self.feat_mean.copy_(mean_t)
            self.feat_std.copy_(std_t)
            self.feat_norm_ready.fill_(1)

    def _normalize_input(self, x):
        """对专家分支输入做逐维标准化；未 fit 时（ready=0）恒等返回。"""
        if int(self.feat_norm_ready.item()) == 1:
            return (x - self.feat_mean) / self.feat_std
        return x

    @staticmethod
    def _otsu_threshold(x, n_bins=50):
        """一维 Otsu 阈值：找使"类间方差"最大的分界点。"""
        x = x.flatten().detach()
        if x.numel() < 2:
            return x.mean() if x.numel() == 1 else torch.tensor(1.5, device=x.device)
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min).item() < 1e-6:
            return x_min

        hist = torch.histc(x, bins=n_bins, min=x_min.item(), max=x_max.item())
        bin_edges = torch.linspace(x_min.item(), x_max.item(), n_bins + 1, device=x.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total = hist.sum()
        if total.item() < 1:
            return x.mean()
        p = hist / total
        omega = torch.cumsum(p, dim=0)
        mu = torch.cumsum(p * bin_centers, 0)
        mu_T = mu[-1]
        denom = omega * (1.0 - omega) + 1e-9
        sigma_B2 = (mu_T * omega - mu).pow(2) / denom
        sigma_B2 = torch.where(torch.isfinite(sigma_B2),
                                sigma_B2,
                                torch.zeros_like(sigma_B2))
        idx = sigma_B2.argmax()
        return bin_centers[idx]

    def _compute_gate(self, x0):
        """计算最终融合 gate ∈ [0, 1]（Otsu + EMA + STE 二值化）。"""
        rel_raw = x0.index_select(1, self._gate_idx_buf)     # (B, 4)
        rel = self.rel_norm(rel_raw)
        prior_offset = torch.norm(rel, dim=1, keepdim=True)   # (B, 1) ||rel||

        if self.training and prior_offset.size(0) >= self.OTSU_MIN_SAMPLES:
            with torch.no_grad():
                batch_thr = self._otsu_threshold(prior_offset, self.OTSU_N_BINS)
                cur_std   = prior_offset.std()
                m = self.EMA_MOMENTUM
                self.threshold_ema.mul_(m).add_(batch_thr, alpha=1.0 - m)
                self.offset_ema_std.mul_(m).add_(cur_std,  alpha=1.0 - m)

        threshold = self.threshold_ema
        sharpness = self.K_SHARP / (self.offset_ema_std + 1e-6)

        soft_gate = torch.sigmoid((prior_offset - threshold) * sharpness)
        hard_gate = (soft_gate > 0.5).float()
        prior_gate = hard_gate + (soft_gate - soft_gate.detach())

        if self.training:
            cnt = getattr(self, "_rel_debug_cnt", 0)
            if cnt % 50 == 0:
                with torch.no_grad():
                    on_ratio = hard_gate.mean().item()
                    batch_thr_val = self._otsu_threshold(
                        prior_offset, self.OTSU_N_BINS).item()
                    print(f"[rel diag] step={cnt} | "
                          f"||rel||: mean={prior_offset.mean().item():.4f} "
                          f"max={prior_offset.max().item():.4f} | "
                          f"otsu_thr_batch={batch_thr_val:.4f} "
                          f"thr_ema={threshold.item():.4f} "
                          f"sharp={sharpness.item():.3f} | "
                          f"ema_std={self.offset_ema_std.item():.4f} | "
                          f"on_ratio={on_ratio:.3f}")
            self._rel_debug_cnt = cnt + 1

        gate_input = torch.cat([rel, rel.abs()], dim=1)       # (B, 8)
        residual = self.gate_residual(gate_input) * self.GATE_RESIDUAL_SCALE
        gate = torch.clamp(prior_gate + residual, 0.0, 1.0)
        return gate, prior_gate

    def forward(self, x):
        x0 = x                          # 原始特征：供门控（Otsu 物理尺度）与 anchor 使用
        xn = self._normalize_input(x0)  # 标准化特征：供两个专家分支使用

        # ============ HEAD 专家：与 IRIS 完全同构，同样看全 124 维（标准化后）============
        h1 = self.head_input_layer(xn)
        h2 = self.head_cross_net(xn)
        h_main = torch.cat([h1, h2], dim=1)
        h_main = self.head_block1(h_main)
        h_main = self.head_block2(h_main)
        h_main = self.head_block3(h_main)
        h_main = self.head_block4(h_main)           # (B, 16)
        y_head = self.head_output_layer(h_main)     # (B, 2)

        # ============ IRIS 专家：主干（标准化后）============
        x1 = self.input_layer(xn)
        x2 = self.cross_net(xn)
        x_main = torch.cat([x1, x2], dim=1)
        x_main = self.block1(x_main)
        x_main = self.block2(x_main)
        x_main = self.block3(x_main)
        x_main = self.block4(x_main)  # (B, 16)

        # 把 y_head 作为 anchor 拼接到 iris 主干特征（detach 避免污染 y_head 的语义）
        x_with_head = torch.cat([x_main, y_head.detach()], dim=1)   # (B, 18)
        y_iris = self.output_layer(x_with_head)     # (B, 2) 虹膜偏转贡献量

        # 门控融合：注意 gate 用【原始特征 x0】，保留 ||rel|| 的物理尺度
        gate, prior_gate = self._compute_gate(x0)
        y_final = y_head + gate * y_iris

        if self.training:
            return y_final, y_iris, y_head, gate, prior_gate
        return y_final


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and skip connection."""

    def __init__(self, in_features, out_features, dropout=0.4):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main(x)
        out = out + residual
        out = self.relu(out)
        return out
