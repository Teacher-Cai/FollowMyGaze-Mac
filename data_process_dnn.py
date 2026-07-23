import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from global_info import GlobalInfo
import threading

# 全局变量：基于 GlobalInfo.path_dir 的绝对路径（打包后 .app 工作目录是 /，不能用相对路径）
SAMPLE_DIR = os.path.join(GlobalInfo.path_dir, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)


class GazeDataset(Dataset):
    def __init__(self, samples_dir=SAMPLE_DIR, ui_callback=None):
        self.samples_dir = samples_dir
        self.ui_callback = ui_callback
        self.features = None
        self.labels = None
        self.sample_count = 0
        self.features_new = []
        self.labels_new = []
        self.df = pd.DataFrame()
        self._lock = threading.Lock()  # 线程安全锁
        # 本地 pkl 已持久化样本数（启动时从文件读取，之后每次 auto-save 更新）
        self.local_sample_count = self._count_local_samples()

    def _count_local_samples(self):
        """启动时统计本地 pkl 已持久化的样本数（不加载全部数据到内存）。"""
        annotations_file = os.path.join(self.samples_dir, 'samples.pkl')
        if not os.path.exists(annotations_file):
            return 0
        try:
            df = pd.read_pickle(annotations_file)
            return len(df)
        except Exception as e:
            print(f"[GazeDataset] count local samples failed: {e}")
            return 0

    def _format_sample_count(self):
        """UI 显示格式：本地X+本次Y=Z（Y = 本次会话中尚未持久化的样本数）。"""
        local = self.local_sample_count
        pending = len(self.df)
        return f"累计样本：本地{local}+本次{pending}={local + pending}"

    def _push_sample_count_to_ui(self):
        if GlobalInfo.sample_count_var is not None and GlobalInfo.root is not None:
            text = self._format_sample_count()
            GlobalInfo.root.after(0, lambda t=text: GlobalInfo.sample_count_var.set(t))

    def __len__(self):
        # 训练时 self.features 是固定的 numpy array，用它的长度防止 DataLoader 越界
        if self.features is not None:
            return len(self.features)
        return len(self.features_new)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_all_samples(self):
        if not os.path.exists(self.samples_dir):
            return
        features = []
        labels = []
        # 读取标注数据
        annotations_file = os.path.join(self.samples_dir, 'samples.pkl')
        if not os.path.exists(annotations_file):
            print("Annotations file not found")
            return

        df = pd.read_pickle(annotations_file)

        for index, row in df.iterrows():
            features.append(row['features'])
            labels.append([row['gaze_x'], row['gaze_y']])

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.sample_count = len(self.features)

    def save_sample(self, features, coords):
        with self._lock:
            self.features_new.append(features)
            self.labels_new.append(coords)

            self.sample_count = len(self.features_new)

            new_row = pd.DataFrame({'features': [features],
                                    'gaze_x': coords[0],
                                    'gaze_y': coords[1]})

            print(new_row)
            self.df = pd.concat([self.df, new_row], axis=0)  # 沿行方向拼接

            if self.ui_callback:
                self.ui_callback("sample_saved", self.sample_count)
            print(f"Sample saved. Session samples: {self.sample_count}, "
                  f"total (local+session): {self.local_sample_count + len(self.df)}")

            # 实时更新 GUI 样本数显示（通过 after 转到主线程，避免 Tkinter 线程安全问题）
            self._push_sample_count_to_ui()

            # 每攒够 auto_train_threshold 个新样本 → 自动持久化 + 自动全量训练
            if self.sample_count > 0 and self.sample_count % GlobalInfo.auto_train_threshold == 0:
                self._auto_save_to_local()
                if GlobalInfo.enable_auto_train:
                    self._trigger_auto_train()

    def _trigger_auto_train(self):
        """后台线程触发一次全量样本训练（warm start，使用现有模型权重继续训练）。

        - 使用当前模型 checkpoint 作为初始化，不冷启动
        - 训练全部本地样本（load_all_samples）
        - 训练在独立线程执行，不阻塞采集主线程
        - 通过 GazeController.training 标志防止重入（若上一轮还没跑完，跳过本次）
        """
        controller = GlobalInfo.train_and_predict_instance
        if controller is None:
            print("[auto-train] skipped: no controller instance")
            return
        if getattr(controller, "training", False):
            print("[auto-train] skipped: previous training still running")
            return

        def _run():
            try:
                print(f"[auto-train] triggered (samples={self.sample_count}, "
                      f"epoch={GlobalInfo.online_training_epoch})")
                # 备份当前采集侧的 sample_count（会被 load_all_samples 覆盖），
                # 训练结束后需要恢复，否则触发条件 % auto_train_threshold 会错乱
                saved_sample_count = self.sample_count
                # 备份当前的 offline_training_epoch，临时改成 online_training_epoch
                saved_epoch = GlobalInfo.offline_training_epoch
                GlobalInfo.offline_training_epoch = GlobalInfo.online_training_epoch
                try:
                    # is_online_mode=False → 走 load_all_samples 全量训练路径
                    # clear_after=False：不要清空 features_new，采集主线程还在往里追加
                    controller.train_model(is_online_mode=False, clear_after=False)
                finally:
                    GlobalInfo.offline_training_epoch = saved_epoch
                    # 恢复采集侧的计数（sample_count 用于触发下一轮 auto-train）
                    self.sample_count = saved_sample_count
                print("[auto-train] finished")
            except Exception as e:
                print(f"[auto-train] error: {e}")

        threading.Thread(target=_run, daemon=True, name="AutoTrainThread").start()

    def _auto_save_to_local(self):
        """将当前累积的样本追加到本地文件，保存后清空 df 避免重复。"""
        annotations_file = os.path.join(self.samples_dir, 'samples.pkl')

        if os.path.exists(annotations_file):
            df_local = pd.read_pickle(annotations_file)
        else:
            df_local = pd.DataFrame()

        df_local = pd.concat([df_local, self.df], axis=0)
        self.df = pd.DataFrame()  # 清空已保存的数据，避免下次重复写入

        # 检查上限
        if len(df_local) > GlobalInfo.sample_upper_limit:
            df_local = df_local[-GlobalInfo.sample_upper_limit:]
            print("over max samples value, remove oldest samples!")

        df_local.to_pickle(annotations_file)
        # 更新本地样本计数（保存后本地总数 = 文件里的行数）
        self.local_sample_count = len(df_local)
        print(f"Auto-saved {GlobalInfo.auto_train_threshold} samples to local. Total local samples: {len(df_local)}")
        # 同步 UI
        self._push_sample_count_to_ui()

    def check_sample_upper_limit(self, df):
        if len(df) > GlobalInfo.sample_upper_limit:
            df = df[-GlobalInfo.sample_upper_limit:]
            print("over max samples value, remove oldest samples!")

    def use_current_samples(self):
        self.features = np.array(self.features_new)
        self.labels = np.array(self.labels_new)
        print(f"Created {self.sample_count} sample images and annotations")

    def clear_samples(self):
        self.features = None
        self.labels = None
        self.sample_count = 0
        self.features_new.clear()
        self.labels_new.clear()

    def save_current_sample_to_local(self):
        """程序退出时调用，保存所有未持久化的样本"""
        with self._lock:
            if self.df.empty:
                return

            if not os.path.exists(self.samples_dir):
                return
            annotations_file = os.path.join(self.samples_dir, 'samples.pkl')
            if not os.path.exists(annotations_file):
                print("Annotations file not found, new one")
                df_local = pd.DataFrame()
            else:
                df_local = pd.read_pickle(annotations_file)

            df_local = pd.concat([df_local, self.df], axis=0)
            self.check_sample_upper_limit(df_local)
            df_local.to_pickle(annotations_file)
            # 同步本地计数（退出前如果 GUI 还在，也能反映一下）
            self.local_sample_count = len(df_local)
            self.df = pd.DataFrame()
            print("save sample completed, total samples:", len(df_local))
