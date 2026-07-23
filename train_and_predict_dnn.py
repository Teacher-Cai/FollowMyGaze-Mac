import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_process_dnn import GazeDataset
from gaze_feature_based_model import GazeMoE
from global_info import GlobalInfo


class GazeController:
    # 多任务监督权重
    LOSS_W_FINAL    = 1.0    # 最终预测的主 loss
    LOSS_W_HEAD_AUX = 0.8    # 仅在虹膜居中时生效，适度保留
    LOSS_W_IRIS_AUX = 0.5    # 提高权重，让 y_iris 获得足够梯度
    LOSS_W_IRIS_ZERO_LAMBDA = 1.0   # iris_aux 内归零项的相对弱化系数（0.1→0.2，限制 y_iris 越界）

    def __init__(self, ui_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GazeMoE().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.ui_callback = ui_callback
        self.training = False
        self.is_predict = False

        # EMA 平滑状态
        self._ema_x = None
        self._ema_y = None
        self._ema_alpha = 0.9  # 平滑系数：越小越平滑但延迟越大

        # Load existing model if exists
        if os.path.exists(os.path.join(GlobalInfo.path_dir, "gaze_model_resnet.pth")):
            # strict=False 兼容模型结构变更（如新增 gate_residual / head_branch 子模块）
            self.model.load_state_dict(
                torch.load(os.path.join(GlobalInfo.path_dir, "gaze_model_resnet.pth"),
                           map_location=self.device),
                strict=False,
            )

    def _compute_loss(self, outputs, targets):
        """多任务监督损失（gate 关系修正版）。

        语义：
          - y_head : 头部姿态贡献的注视点位置（base 预测）
          - y_iris : 虹膜偏转贡献的位移量
          - y_final = y_head + gate * y_iris

        三个 loss 的 gate 加权逻辑（按物理直觉）：
          - gate ≈ 0 (虹膜居中)：label ≈ 头部姿态决定的位置
                · 应强监督 y_head ≈ label（这是干净标签）
                · 应强制 y_iris ≈ 0（虹膜没动，不该有贡献）
          - gate ≈ 1 (虹膜偏转)：label 包含虹膜贡献
                · 应弱监督 y_head（避免污染）
                · 应让 y_iris 学到完整的虹膜贡献量

        - loss_final    : MSE(y_head + gate*y_iris, label)              — 主目标
        - loss_head_aux : (1-gate.detach()) * (y_head - label)^2        — 虹膜居中时监督 y_head
        - loss_iris_aux : gate.detach() * (y_head.detach()+y_iris-label)^2
                          + (1-gate.detach()) * y_iris^2
                          — 虹膜偏转时学贡献量，虹膜居中时归零
        """
        y_final, y_iris, y_head, gate, prior_gate = outputs

        loss_final = self.criterion(y_final, targets)

        gate_d = gate.detach()                                   # (B, 1)
        prior_d = prior_gate.detach()                            # (B, 1)
        residual_abs = (gate_d - prior_d).abs()                  # 检查 residual 贡献

        # head_aux：仅在 gate 小（虹膜居中）时强监督 y_head
        head_err = (y_head - targets).pow(2)                     # (B, 2)
        loss_head_aux = ((1.0 - gate_d) * head_err + gate_d * head_err * 0.5).mean()  #

        # iris_aux：gate 大时学贡献量，gate 小时归零（归零项被 ZERO_LAMBDA 弱化、避免压死修正项）
        err_contrib = (y_head.detach() + y_iris - targets).pow(2)
        err_zero    = y_iris.pow(2)
        loss_iris_aux = (gate_d * err_contrib
                         + self.LOSS_W_IRIS_ZERO_LAMBDA * (1.0 - gate_d) * err_zero).mean()

        total = (self.LOSS_W_FINAL * loss_final
                 + self.LOSS_W_HEAD_AUX * loss_head_aux
                 + self.LOSS_W_IRIS_AUX * loss_iris_aux)
        return total, {
            'final': loss_final.item(),
            'head_aux': loss_head_aux.item(),
            'iris_aux': loss_iris_aux.item(),
            'gate_mean': gate_d.mean().item(),
            'gate_min':  gate_d.min().item(),
            'gate_max':  gate_d.max().item(),
            'gate_q25':  gate_d.quantile(0.25).item(),
            'gate_q75':  gate_d.quantile(0.75).item(),
            'prior_gate_mean': prior_d.mean().item(),
            'gate_residual_abs_mean': residual_abs.mean().item(),
            'y_iris_abs_mean': y_iris.detach().abs().mean().item(),
        }

    def train_model(self, is_online_mode=True, clear_after=True):
        if self.training:
            return
        self.training = True
        if self.ui_callback:
            self.ui_callback("training_started")
        print("Starting model training...")
        dataset = GlobalInfo.train_data
        if is_online_mode:
            dataset.use_current_samples()
        else:
            dataset.load_all_samples()

        training_batch_size = GlobalInfo.online_training_batchSize if is_online_mode else GlobalInfo.offline_training_batchSize

        dataloader = DataLoader(dataset, batch_size=training_batch_size, shuffle=True)

        # —— 影子模型 shadow：训练不动 self.model，避免与推理线程抢模型状态 ——
        # 从当前推理模型 warm start，训练完再原子替换 self.model
        # 这样根治了：1) forward 返回值元数不一致（train/eval 被推理线程切换）
        #             2) BatchNorm running stats 训练中被推理污染
        #             3) 推理帧读到"半成品权重"抖动
        shadow = GazeMoE().to(self.device)
        shadow.load_state_dict(self.model.state_dict())
        shadow.train()

        # —— 拟合输入特征标准化统计量（mean/std）——
        # 数据量纲差异极大，标准化后专家分支训练更稳、精度更高。
        # 策略：offline（全量样本，权威）总是重新拟合；online（仅少量新样本，噪声大）
        #      仅当模型尚未初始化统计量时才拟合，避免每次在线训练来回漂移。
        try:
            feats = getattr(dataset, "features", None)
            if feats is not None and len(feats) >= 2:
                need_fit = (not is_online_mode) or \
                           (int(shadow.feat_norm_ready.item()) == 0)
                if need_fit:
                    feats_t = torch.as_tensor(np.asarray(feats), dtype=torch.float32)
                    mean = feats_t.mean(dim=0)
                    std = feats_t.std(dim=0)
                    shadow.set_feature_stats(mean, std)
                    print(f"[feat-norm] fitted on {len(feats)} samples "
                          f"(mean|std example: dim0 mean={mean[0]:.3f} std={std[0]:.3f})")
        except Exception as e:
            print(f"[feat-norm] fit skipped due to: {e}")

        shadow_optim = optim.Adam(shadow.parameters(),
                                  lr=self.optimizer.param_groups[0]['lr'])

        training_epoch = GlobalInfo.online_training_epoch if is_online_mode else GlobalInfo.offline_training_epoch
        for epoch in range(training_epoch):
            total_loss = 0
            loss_parts_sum = {'final': 0.0, 'head_aux': 0.0, 'iris_aux': 0.0,
                              'gate_mean': 0.0, 'gate_min': 0.0, 'gate_max': 0.0,
                              'gate_q25': 0.0, 'gate_q75': 0.0,
                              'prior_gate_mean': 0.0, 'gate_residual_abs_mean': 0.0,
                              'y_iris_abs_mean': 0.0}
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                targets = targets.float()

                shadow_optim.zero_grad()
                images = torch.FloatTensor(images.float())
                outputs = shadow(images)   # 训练模式下返回 (y_final, y_iris, y_head, gate, prior_gate)
                loss, parts = self._compute_loss(outputs, targets)
                loss.backward()
                shadow_optim.step()

                total_loss += loss.item()
                for k in loss_parts_sum:
                    loss_parts_sum[k] += parts[k]

            n = len(dataloader)
            print(f"Epoch [{epoch + 1}/{training_epoch}], Loss: {total_loss / n:.4f} "
                  f"(final={loss_parts_sum['final']/n:.2f}, "
                  f"head_aux={loss_parts_sum['head_aux']/n:.2f}, "
                  f"iris_aux={loss_parts_sum['iris_aux']/n:.2f}, "
                  f"|y_iris|={loss_parts_sum['y_iris_abs_mean']/n:.2f})")
            print(f"  gate stats: mean={loss_parts_sum['gate_mean']/n:.3f}, "
                  f"min={loss_parts_sum['gate_min']/n:.3f}, "
                  f"max={loss_parts_sum['gate_max']/n:.3f}, "
                  f"q25={loss_parts_sum['gate_q25']/n:.3f}, "
                  f"q75={loss_parts_sum['gate_q75']/n:.3f}, "
                  f"prior_mean={loss_parts_sum['prior_gate_mean']/n:.3f}, "
                  f"|residual|={loss_parts_sum['gate_residual_abs_mean']/n:.3f}")
            if self.ui_callback:
                self.ui_callback("training_progress", epoch + 1, total_loss / n)

        # 训练完成：切 eval，持久化，然后原子替换推理模型
        shadow.eval()
        torch.save(shadow.state_dict(), os.path.join(GlobalInfo.path_dir, "gaze_model_resnet.pth"))
        # Python 属性赋值本身是原子的：推理下一帧就会用新模型
        self.model = shadow
        self.optimizer = shadow_optim
        print("Model training completed.")
        self.training = False
        if clear_after:
            dataset.clear_samples()

    def predict_gaze(self, features):
        if self.is_predict:
            return
        self.is_predict = True
        try:
            with torch.no_grad():
                self.model.eval()  # eval 模式下 forward 只返回 y_final
                feature_tensor = torch.FloatTensor(features.reshape(1, -1))
                image = feature_tensor.to(self.device)
                output = self.model(image)
                coords = output.cpu().numpy()[0]

                raw_x = float(coords[0])
                raw_y = float(coords[1])

                # EMA 平滑：减少帧间抖动
                if self._ema_x is None:
                    self._ema_x, self._ema_y = raw_x, raw_y
                else:
                    self._ema_x = self._ema_alpha * raw_x + (1 - self._ema_alpha) * self._ema_x
                    self._ema_y = self._ema_alpha * raw_y + (1 - self._ema_alpha) * self._ema_y

                x = int(self._ema_x)
                y = int(self._ema_y)
                return x, y
        finally:
            self.is_predict = False


if __name__ == "__main__":
    GlobalInfo.train_data = GazeDataset('/Users/caijiawei/FollowMyGaze/samples')
    aaa = GazeController()
    aaa.train_model(is_online_mode=False)
