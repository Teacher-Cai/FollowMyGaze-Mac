"""GUI 主模块。

设计原则：
- 所有 side effect（Tk 实例、控件、监听线程）都封装在 run_gui() 中，
  import 本模块不会自动启动 UI，方便测试和被脚本引用。
- 摄像头帧循环 & MediaPipe 特征提取都跑在后台线程池，主线程只做 UI 更新。
- 训练相关按钮/自动训练全部走后台线程，绝不阻塞 Tkinter mainloop。
"""

import logging
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

import utils
from gaze_cursor_modes import CursorModeManager
from global_info import GlobalInfo
from user_config import load_gui_config, save_gui_config

logger = logging.getLogger(__name__)

# —— 视频循环参数 ——
_FRAME_INTERVAL_MS = 40            # 目标 ~30fps
_CAMERA_FAIL_TIMEOUT_SEC = 3.0     # 连续读帧失败超过此秒数则触发重连
_MAX_CAMERA_FAILS = int(_CAMERA_FAIL_TIMEOUT_SEC * 1000 / _FRAME_INTERVAL_MS)

# —— 视频显示的初始尺寸（会随窗口大小动态调整） ——
_INITIAL_VIDEO_W = 600
_INITIAL_VIDEO_H = 500

# —— 红点参数 ——
_RED_DOT_SIZE = 20
_RED_DOT_REFRESH_MS = 40           # 位置刷新间隔（≈33fps 跟随预测）


class RedDotOverlay:
    """持久的红点悬浮窗（全屏透明层 + 单个红点圆形）。

    - start(): 创建 Toplevel + Canvas + 圆形；启动位置刷新循环
    - stop() : 销毁窗口，停止刷新循环
    - 位置由 GlobalInfo.red_dot_x / red_dot_y 驱动，_tick 每 30ms 移动圆形
    - 幂等：重复 start/stop 无副作用
    """

    def __init__(self, root):
        self._root = root
        self._win = None       # tk.Toplevel
        self._canvas = None    # tk.Canvas
        self._dot_id = None    # canvas item id
        self._active = False   # 逻辑开关；stop 后 _tick 自动退出
        self._tick_scheduled = False

    def is_active(self):
        return self._active

    def start(self):
        if self._active:
            return
        try:
            self._win = tk.Toplevel(self._root)
            self._win.title("红点追踪（按 Esc 关闭）")
            self._win.attributes("-topmost", True)

            w = GlobalInfo.screen_width or self._win.winfo_screenwidth()
            h = GlobalInfo.screen_height or self._win.winfo_screenheight()
            # 用 geometry 覆盖全屏，但**保留标题栏**（方便关闭）
            self._win.geometry(f"{w}x{h}+0+0")

            bg_color = "white"
            alpha = 1.0
            self._win.attributes("-alpha", alpha)
            self._win.attributes("-fullscreen", True)

            # 关闭方式 1：窗口右上角 X（现在有标题栏，可点）
            self._win.protocol('WM_DELETE_WINDOW', self.stop)
            # 关闭方式 2：Esc 键
            self._win.bind('<Escape>', lambda e: self.stop())
            # 让窗口能接收键盘焦点
            self._win.focus_force()

            self._canvas = tk.Canvas(self._win, width=w, height=h,
                                      bg=bg_color, highlightthickness=0)
            self._canvas.pack()
            # 关闭方式 3：点击红点/画布
            self._canvas.bind('<Button-1>', lambda e: self.stop())

            self._dot_id = self._canvas.create_oval(
                0, 0, _RED_DOT_SIZE, _RED_DOT_SIZE, fill="red", outline="")
            self._active = True
            self._schedule_tick()
            logger.info("RedDotOverlay started")
        except Exception:
            logger.exception("RedDotOverlay.start failed")
            self._cleanup()

    def stop(self):
        if not self._active and self._win is None:
            return
        self._active = False
        self._cleanup()
        logger.info("RedDotOverlay stopped")

    def _cleanup(self):
        try:
            if self._win is not None:
                self._win.destroy()
        except Exception:
            pass
        self._win = None
        self._canvas = None
        self._dot_id = None

    def _schedule_tick(self):
        if self._active and not self._tick_scheduled:
            self._tick_scheduled = True
            self._root.after(_RED_DOT_REFRESH_MS, self._tick)

    def _tick(self):
        self._tick_scheduled = False
        if not self._active or self._canvas is None or self._dot_id is None:
            return
        try:
            x = GlobalInfo.red_dot_x
            y = GlobalInfo.red_dot_y
            # 直接把圆形 item 移到新位置（比 delete+create 快很多，无闪烁）
            self._canvas.coords(self._dot_id,
                                x, y, x + _RED_DOT_SIZE, y + _RED_DOT_SIZE)
        except Exception:
            logger.exception("RedDotOverlay._tick failed")
        finally:
            self._schedule_tick()


class GazeApp:
    """所有 UI 状态封装到实例，避免模块级 global。

    仅由 run_gui() 构造一次。
    """

    def __init__(self):
        # ---------- 根窗口 ----------
        self.root = tk.Tk()
        self.root.title("看我眼神")
        self.root.geometry("800x760+0+0")
        self.root.resizable(True, True)
        GlobalInfo.root = self.root

        # —— 设置窗口图标 ——
        try:
            from utils import resource_path
            icon_img = ImageTk.PhotoImage(Image.open(resource_path("icon.png")))
            self.root.iconphoto(True, icon_img)
        except Exception:
            pass

        # ---------- 后台线程池 ----------
        # 一个 worker 处理"特征提取 + 推理"，避免主线程被 MediaPipe 阻塞
        self._frame_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='frame')
        GlobalInfo.predict_executor = self._frame_executor
        self._frame_future = None

        # ---------- 视频区 ----------
        self.video_label = tk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._video_w = _INITIAL_VIDEO_W
        self._video_h = _INITIAL_VIDEO_H
        self.video_label.bind("<Configure>", self._on_video_resize)

        # ---------- 信息条 ----------
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=6)

        tk.Label(info_frame, text="提示信息：").grid(row=0, column=0)
        self.model_state_str = tk.StringVar(value="……")
        tk.Label(info_frame, textvariable=self.model_state_str,
                 width=16, anchor='w').grid(row=0, column=1)
        GlobalInfo.model_state_var = self.model_state_str

        self.sample_count_str = tk.StringVar(value="累计样本：本地0+本次0=0")
        tk.Label(info_frame, textvariable=self.sample_count_str,
                 width=32, anchor='w').grid(row=0, column=2, padx=(20, 0))
        GlobalInfo.sample_count_var = self.sample_count_str

        self.mode_status_str = tk.StringVar(value="当前模式：后台训练")
        tk.Label(info_frame, textvariable=self.mode_status_str,
                 width=18).grid(row=0, column=3, padx=(20, 0))

        # ---------- 模式单选（红点已改为独立开关，不再在这里） ----------
        self._build_mode_selector()

        # ---------- 红点悬浮层（持久）----------
        self.red_dot_overlay = RedDotOverlay(self.root)

        # ---------- 视线-鼠标交互模式管理器 ----------
        self.cursor_mode_manager = CursorModeManager()

        # ---------- 开关：红点显示 / 自动训练 ----------
        self._build_switches()

        # ---------- 训练按钮 + 进度条 ----------
        self._build_train_area()

        # ---------- 摄像头恢复计数 ----------
        self._camera_fail_count = 0

        # 关闭协议
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ================== 界面构造 ==================
    def _build_mode_selector(self):
        style = ttk.Style()
        style.configure("Big.TRadiobutton",
                        font=("Arial", 9), padding=5, indicatorsize=15)

        frame = tk.Frame(self.root)
        frame.pack()
        tk.Label(frame, text="模式选择：").grid(row=0, column=0)
        self.which_mode = tk.StringVar(value='silent_train')
        GlobalInfo.mode_select = self.which_mode

        ttk.Radiobutton(frame, value="silent_train", variable=self.which_mode,
                        text="后台训练", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=1)
        ttk.Radiobutton(frame, value='move_cursor', variable=self.which_mode,
                        text="光标随动(Alt)", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=2)
        ttk.Radiobutton(frame, value='gaze_jump', variable=self.which_mode,
                        text="视线跳转", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=3)
        ttk.Radiobutton(frame, value='gaze_follow', variable=self.which_mode,
                        text="视线跟随", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=4)
        ttk.Radiobutton(frame, value='gaze_glide', variable=self.which_mode,
                        text="视线滑翔", style="Big.TRadiobutton",
                        command=self._on_mode_changed).grid(row=0, column=5)

    def _build_switches(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=6)

        # 红点按钮：点击弹出红点追踪窗口；用户关掉那个窗口即结束
        self.red_dot_btn = tk.Button(frame, text="显示红点追踪",
                                     width=16,
                                     command=self._on_red_dot_open)
        self.red_dot_btn.grid(row=0, column=0, padx=8)

        # auto-train 开关（与 GlobalInfo.enable_auto_train 同步）
        self.auto_train_var = tk.BooleanVar(value=bool(GlobalInfo.enable_auto_train))
        tk.Checkbutton(frame, text=f"自动训练（每 {GlobalInfo.auto_train_threshold} 样本）",
                       variable=self.auto_train_var,
                       command=self._on_auto_train_toggle
                       ).grid(row=0, column=1, padx=8)

        # 摄像头选择（多摄像头设备可切换；切换后立即重连生效）
        tk.Label(frame, text="摄像头：").grid(row=0, column=2, padx=(20, 2))
        self.camera_index_var = tk.StringVar(
            value=str(int(getattr(GlobalInfo, 'camera_index', 0))))
        self.camera_combo = ttk.Combobox(
            frame, width=4, state='normal',
            values=[str(i) for i in range(6)],  # 0~5，也可手动输入更大索引
            textvariable=self.camera_index_var)
        self.camera_combo.grid(row=0, column=3)
        # 下拉选中 / 回车确认 均触发切换
        self.camera_combo.bind('<<ComboboxSelected>>', self._on_camera_index_changed)
        self.camera_combo.bind('<Return>', self._on_camera_index_changed)
        self.camera_hint = tk.Label(
            frame, text=f"当前：{int(getattr(GlobalInfo, 'camera_index', 0))}",
            fg="green", width=8, anchor='w')
        self.camera_hint.grid(row=0, column=4, padx=(4, 0))

        # ─── 视线跳转 ───
        jump_frame = tk.LabelFrame(self.root, text="视线跳转", padx=6, pady=4)
        jump_frame.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(jump_frame, text="触发距离(px)：").grid(row=0, column=0, sticky='w')
        self.jump_threshold_var = tk.StringVar(
            value=str(int(GlobalInfo.gaze_jump_jump_threshold)))
        # 每次内容变化立即回调
        self.jump_threshold_var.trace_add(
            'write', lambda *_: self._on_jump_threshold_changed())
        self.jump_threshold_entry = tk.Entry(
            jump_frame, textvariable=self.jump_threshold_var,
            width=8, justify='right')
        self.jump_threshold_entry.grid(row=0, column=1, padx=6)
        tk.Label(jump_frame, text="px").grid(row=0, column=2)
        # 显示当前生效值（非法输入时可以看到 UI 与生效值的差异）
        self.jump_threshold_hint = tk.Label(
            jump_frame, text=f"已生效：{int(GlobalInfo.gaze_jump_jump_threshold)} px",
            fg="green", width=20, anchor='w')
        self.jump_threshold_hint.grid(row=0, column=3, padx=(10, 0))

        # 冷却时间
        tk.Label(jump_frame, text="冷却时间(ms)：").grid(row=1, column=0, sticky='w')
        self.jump_cooldown_var = tk.IntVar(
            value=int(GlobalInfo.gaze_jump_cooldown_ms))
        tk.Scale(jump_frame, from_=500, to=5000, resolution=100,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.jump_cooldown_var,
                 command=self._on_jump_cooldown_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.jump_cooldown_hint = tk.Label(
            jump_frame, text=f"当前：{int(GlobalInfo.gaze_jump_cooldown_ms)} ms",
            fg="green", width=14, anchor='w')
        self.jump_cooldown_hint.grid(row=1, column=2, sticky='w')

        # ─── 视线跟随 ───
        follow_frame = tk.LabelFrame(self.root, text="视线跟随", padx=6, pady=4)
        follow_frame.pack(fill='x', padx=8, pady=(0, 6))
        # 用户操作后暂停多久恢复自动跟随
        tk.Label(follow_frame, text="操作后暂停(秒)：").grid(row=0, column=0, sticky='w')
        self.follow_idle_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_follow_idle_seconds))
        tk.Scale(follow_frame, from_=1.0, to=10.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.follow_idle_var,
                 command=self._on_follow_idle_changed,
                 showvalue=False).grid(row=0, column=1, padx=6)
        self.follow_idle_hint = tk.Label(
            follow_frame, text=f"当前：{float(GlobalInfo.gaze_follow_idle_seconds):.1f}s",
            fg="green", width=14, anchor='w')
        self.follow_idle_hint.grid(row=0, column=2, sticky='w')
        # 顺滑度
        tk.Label(follow_frame, text="跟随顺滑度：").grid(row=1, column=0, sticky='w')
        self.follow_ease_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_follow_ease))
        tk.Scale(follow_frame, from_=0.10, to=0.80, resolution=0.05,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.follow_ease_var,
                 command=self._on_follow_ease_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.follow_ease_hint = tk.Label(
            follow_frame,
            text=f"当前：{float(GlobalInfo.gaze_follow_ease):.2f}（小=顺滑）",
            fg="green", width=18, anchor='w')
        self.follow_ease_hint.grid(row=1, column=2, sticky='w')

        # ─── 视线滑翔 ───
        glide_frame = tk.LabelFrame(self.root, text="视线滑翔", padx=6, pady=4)
        glide_frame.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(glide_frame, text="加速倍数：").grid(row=0, column=0, sticky='w')
        self.glide_mul_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_glide_max_multiplier))
        self.glide_mul_scale = tk.Scale(
            glide_frame, from_=5.0, to=30.0, resolution=1.0,
            orient=tk.HORIZONTAL, length=240,
            variable=self.glide_mul_var,
            command=self._on_glide_mul_changed, showvalue=False)
        self.glide_mul_scale.grid(row=0, column=1, padx=6)
        self.glide_mul_hint = tk.Label(
            glide_frame,
            text=f"当前：{GlobalInfo.gaze_glide_max_multiplier:.1f}×",
            fg="green", width=14, anchor='w')
        self.glide_mul_hint.grid(row=0, column=2, padx=(6, 0))

        # 减速起始距离
        tk.Label(glide_frame, text="减速起始距离(px)：").grid(row=1, column=0, sticky='w')
        self.glide_near_var = tk.IntVar(
            value=int(GlobalInfo.gaze_glide_near_threshold))
        tk.Scale(glide_frame, from_=100, to=500, resolution=20,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_near_var,
                 command=self._on_glide_near_changed,
                 showvalue=False).grid(row=1, column=1, padx=6)
        self.glide_near_hint = tk.Label(
            glide_frame, text=f"当前：{int(GlobalInfo.gaze_glide_near_threshold)} px",
            fg="green", width=14, anchor='w')
        self.glide_near_hint.grid(row=1, column=2, sticky='w')
        # 全速加速距离
        tk.Label(glide_frame, text="全速加速距离(px)：").grid(row=2, column=0, sticky='w')
        self.glide_far_var = tk.IntVar(
            value=int(GlobalInfo.gaze_glide_far_threshold))
        tk.Scale(glide_frame, from_=300, to=1000, resolution=50,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_far_var,
                 command=self._on_glide_far_changed,
                 showvalue=False).grid(row=2, column=1, padx=6)
        self.glide_far_hint = tk.Label(
            glide_frame, text=f"当前：{int(GlobalInfo.gaze_glide_far_threshold)} px",
            fg="green", width=14, anchor='w')
        self.glide_far_hint.grid(row=2, column=2, sticky='w')
        # 减速陡峭度
        tk.Label(glide_frame, text="减速陡峭度：").grid(row=3, column=0, sticky='w')
        self.glide_exp_var = tk.DoubleVar(
            value=float(GlobalInfo.gaze_glide_dist_exponent))
        tk.Scale(glide_frame, from_=1.0, to=4.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=240,
                 variable=self.glide_exp_var,
                 command=self._on_glide_exp_changed,
                 showvalue=False).grid(row=3, column=1, padx=6)
        self.glide_exp_hint = tk.Label(
            glide_frame,
            text=f"当前：{float(GlobalInfo.gaze_glide_dist_exponent):.1f}（大=急刹车）",
            fg="green", width=20, anchor='w')
        self.glide_exp_hint.grid(row=3, column=2, sticky='w')

        # 控件全部构建完成后，加载上次保存的 GUI 配置并恢复
        self._load_gui_config()

    def _load_gui_config(self):
        """从本地读取上次保存的 GUI 配置，回填各控件。

        注意：模式（radio button）不做持久化恢复，每次启动都由用户重新选择，
        以避免自动创建 pynput Listener 引发 macOS Quartz 懒加载问题。
        """
        cfg = load_gui_config()
        if not cfg:
            return
        try:
            if "auto_train" in cfg:
                self.auto_train_var.set(bool(cfg["auto_train"]))
                self._on_auto_train_toggle()
        except Exception:
            logger.exception("restore auto_train failed")
        try:
            if "camera_index" in cfg:
                idx = int(cfg["camera_index"])
                # main.py 启动时已用该值打开摄像头，这里只同步 UI 显示；
                # 若摄像头尚未就绪（例如启动时设备被占用），则触发一次重连
                self.camera_index_var.set(str(idx))
                if idx != GlobalInfo.camera_index or \
                        GlobalInfo.video_steam is None or \
                        not GlobalInfo.video_steam.isOpened():
                    self._on_camera_index_changed()
                else:
                    self.camera_hint.config(text=f"当前：{idx}", fg="green")
        except Exception:
            logger.exception("restore camera_index failed")
        try:
            if "jump_threshold" in cfg:
                self.jump_threshold_var.set(str(int(cfg["jump_threshold"])))
        except Exception:
            logger.exception("restore jump_threshold failed")
        try:
            if "glide_multiplier" in cfg:
                v = float(cfg["glide_multiplier"])
                self.glide_mul_var.set(v)
                self._on_glide_mul_changed(v)
        except Exception:
            logger.exception("restore glide_multiplier failed")

        # ---- 新增：进阶参数 ----
        try:
            if "jump_cooldown_ms" in cfg:
                v = int(cfg["jump_cooldown_ms"])
                self.jump_cooldown_var.set(v)
                self._on_jump_cooldown_changed(v)
        except Exception:
            logger.exception("restore jump_cooldown_ms failed")
        try:
            if "follow_idle_seconds" in cfg:
                v = float(cfg["follow_idle_seconds"])
                self.follow_idle_var.set(v)
                self._on_follow_idle_changed(v)
        except Exception:
            logger.exception("restore follow_idle_seconds failed")
        try:
            if "follow_ease" in cfg:
                v = float(cfg["follow_ease"])
                self.follow_ease_var.set(v)
                self._on_follow_ease_changed(v)
        except Exception:
            logger.exception("restore follow_ease failed")
        try:
            if "glide_near_threshold" in cfg:
                v = int(cfg["glide_near_threshold"])
                self.glide_near_var.set(v)
                self._on_glide_near_changed(v)
        except Exception:
            logger.exception("restore glide_near_threshold failed")
        try:
            if "glide_far_threshold" in cfg:
                v = int(cfg["glide_far_threshold"])
                self.glide_far_var.set(v)
                self._on_glide_far_changed(v)
        except Exception:
            logger.exception("restore glide_far_threshold failed")
        try:
            if "glide_dist_exponent" in cfg:
                v = float(cfg["glide_dist_exponent"])
                self.glide_exp_var.set(v)
                self._on_glide_exp_changed(v)
        except Exception:
            logger.exception("restore glide_dist_exponent failed")

    def _on_glide_mul_changed(self, value):
        """视线滑翔加速倍数滑块回调：实时写回 GlobalInfo。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_glide_max_multiplier = v
        self.glide_mul_hint.config(text=f"当前：{v:.1f}×")

    # ---- 新增：进阶滑块回调（实时生效） ----
    def _on_jump_cooldown_changed(self, value):
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_jump_cooldown_ms = v
        self.jump_cooldown_hint.config(text=f"当前：{v} ms")

    def _on_follow_idle_changed(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_follow_idle_seconds = v
        self.follow_idle_hint.config(text=f"当前：{v:.1f}s")

    def _on_follow_ease_changed(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_follow_ease = v
        self.follow_ease_hint.config(text=f"当前：{v:.2f}（小=顺滑）")

    def _on_glide_near_changed(self, value):
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_glide_near_threshold = v
        self.glide_near_hint.config(text=f"当前：{v} px")

    def _on_glide_far_changed(self, value):
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_glide_far_threshold = v
        self.glide_far_hint.config(text=f"当前：{v} px")

    def _on_glide_exp_changed(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        GlobalInfo.gaze_glide_dist_exponent = v
        self.glide_exp_hint.config(text=f"当前：{v:.1f}（大=急刹车）")

    def _build_train_area(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.train_button = tk.Button(frame, text='用全部数据训练',
                                      command=self._on_train_button)
        self.train_button.grid(row=0, column=0, padx=8)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(frame, orient='horizontal',
                                            length=280, mode='determinate',
                                            maximum=100.0,
                                            variable=self.progress_var)
        self.progress_bar.grid(row=0, column=1, padx=8)

        self.progress_text = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.progress_text, width=22
                 ).grid(row=0, column=2)

    # ================== 事件回调 ==================
    def _on_mode_changed(self):
        val = self.which_mode.get()
        label = {
            'silent_train': '后台训练',
            'move_cursor': '按键模式(Alt)',
            'gaze_jump': '跳转模式',
            'gaze_follow': '跟随模式',
            'gaze_glide': '丝滑模式',
        }.get(val, val)
        self.mode_status_str.set(f"当前模式：{label}")
        # 切换鼠标交互模式（只有 gaze_jump / gaze_follow 会激活 controller）
        try:
            self.cursor_mode_manager.switch_to(val)
        except Exception:
            logger.exception("switch cursor mode failed")

    def _on_red_dot_open(self):
        """点击按钮 → 弹出红点追踪窗口（幂等，已开则忽略）；
        关闭那个弹窗即结束追踪。"""
        self.red_dot_overlay.start()

    def _on_auto_train_toggle(self):
        GlobalInfo.enable_auto_train = bool(self.auto_train_var.get())
        self.model_state_str.set(
            "自动训练已开启" if GlobalInfo.enable_auto_train else "自动训练已关闭")

    def _on_camera_index_changed(self, *_):
        """摄像头选择变化 → 立即重连对应设备。

        - 合法非负整数：写入 GlobalInfo.camera_index 并调用 _reinit_camera 热切换
        - 非法输入：不动 GlobalInfo，hint 变灰提示
        重连结果（成功/失败）由 _reinit_camera 通过 model_state_str 反馈。
        """
        raw = self.camera_index_var.get().strip()
        try:
            idx = int(raw)
            if idx < 0:
                raise ValueError("must be >= 0")
        except (TypeError, ValueError):
            self.camera_hint.config(
                text=f"无效，仍用：{int(GlobalInfo.camera_index)}", fg="gray")
            return
        if idx == GlobalInfo.camera_index and \
                GlobalInfo.video_steam is not None and GlobalInfo.video_steam.isOpened():
            self.camera_hint.config(text=f"当前：{idx}", fg="green")
            return
        GlobalInfo.camera_index = idx
        self.camera_hint.config(text=f"当前：{idx}", fg="green")
        self.model_state_str.set(f"正在切换摄像头{idx}…")
        self._reinit_camera()

    def _on_jump_threshold_changed(self, value=None):
        """输入框内容变化即生效：
        - 合法正整数 → 立即写入 GlobalInfo，hint 显示绿色"已生效"
        - 空 / 非法 → 保留 GlobalInfo 上一个值不变，hint 变灰提示
        （不强制回填输入框，避免用户输入过程被打断——比如输入 "1000" 中间的 "1"）"""
        raw = self.jump_threshold_var.get().strip()
        if not raw:
            self.jump_threshold_hint.config(text="等待输入…", fg="gray")
            return
        try:
            v = int(raw)
            if v <= 0:
                raise ValueError("must be positive")
        except (TypeError, ValueError):
            self.jump_threshold_hint.config(
                text=f"无效，仍用：{int(GlobalInfo.gaze_jump_jump_threshold)} px",
                fg="red")
            return
        GlobalInfo.gaze_jump_jump_threshold = v
        self.jump_threshold_hint.config(text=f"已生效：{v} px", fg="green")

    def _on_video_resize(self, event):
        # 保持一定 padding，避免边缘裁掉
        w = max(120, event.width - 10)
        h = max(90, event.height - 10)
        self._video_w = w
        self._video_h = h

    # ================== 训练按钮 ==================
    def _on_train_button(self):
        controller = GlobalInfo.train_and_predict_instance
        if controller is None:
            self.model_state_str.set("模型未就绪")
            return
        if getattr(controller, "training", False):
            self.model_state_str.set("训练进行中，请稍候…")
            return

        # 注入 ui_callback，让 train_model 回调进度
        prev_cb = controller.ui_callback
        controller.ui_callback = self._training_ui_callback

        def _run():
            try:
                self._set_train_button_state(disabled=True)
                self.progress_var.set(0.0)
                self.progress_text.set("准备中…")
                self.model_state_str.set("训练中…")
                controller.train_model(is_online_mode=False)
                self.root.after(0, lambda: self.progress_var.set(100.0))
                self.root.after(0, lambda: self.progress_text.set("已完成"))
                self.root.after(0, lambda: self.model_state_str.set("训练完成"))
            except Exception as e:
                logger.exception("manual train failed")
                self.root.after(0, lambda: self.model_state_str.set(f"训练失败：{e}"))
                self.root.after(0, lambda: self.progress_text.set("失败"))
            finally:
                controller.ui_callback = prev_cb
                self._set_train_button_state(disabled=False)

        threading.Thread(target=_run, daemon=True, name="ManualTrainThread").start()

    def _set_train_button_state(self, disabled: bool):
        state = 'disabled' if disabled else 'normal'
        self.root.after(0, lambda: self.train_button.config(state=state))

    def _training_ui_callback(self, event, *args):
        """train_model 回调，跨线程更新 UI 需 root.after(0, ...)。"""
        if event == "training_started":
            self.root.after(0, lambda: self.progress_text.set("开始训练…"))
        elif event == "training_progress":
            # args = (epoch, loss)
            epoch, loss = args[0], args[1]
            total = GlobalInfo.offline_training_epoch
            pct = min(100.0, epoch / max(1, total) * 100.0)
            self.root.after(0, lambda p=pct, e=epoch, t=total, l=loss:
                            (self.progress_var.set(p),
                             self.progress_text.set(f"epoch {e}/{t}  loss={l:.2f}")))

    # ================== 摄像头帧循环 ==================
    def _reinit_camera(self):
        self._camera_fail_count = 0
        cam_idx = int(getattr(GlobalInfo, 'camera_index', 0))
        try:
            if GlobalInfo.video_steam is not None:
                GlobalInfo.video_steam.release()
        except Exception:
            logger.exception("release camera failed")
        try:
            GlobalInfo.video_steam = cv2.VideoCapture(cam_idx)
            if GlobalInfo.video_steam.isOpened():
                GlobalInfo.video_steam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("camera %d reinitialized successfully", cam_idx)
                self.model_state_str.set(f"摄像头{cam_idx}已连接")
            else:
                logger.warning("camera %d reinit failed, will retry", cam_idx)
                self.model_state_str.set(f"摄像头{cam_idx}连接中...")
        except Exception:
            logger.exception("camera reinit exception")
            self.model_state_str.set("摄像头异常")

    def _process_frame_bg(self, frame):
        """后台线程：特征提取 + 推理。不返回，结果通过 GlobalInfo 传递。"""
        try:
            extractor = GlobalInfo.gaze_feature_extractor
            if extractor is None:
                return
            features = extractor.extract_features_from_image(frame)
            GlobalInfo.current_features = features
            utils.predict_and_draw_pot(features)
        except Exception:
            logger.exception("frame background task failed")

    def _update_video_feed(self):
        try:
            # 摄像头未就绪保护
            if GlobalInfo.video_steam is None:
                self.model_state_str.set("摄像头未初始化")
                self.root.after(500, self._update_video_feed)
                return

            ret, frame = GlobalInfo.video_steam.read()
            if ret:
                self._camera_fail_count = 0
                GlobalInfo.current_frame = frame

                # 提交后台：特征提取 + 推理；上一帧未完成则跳过，防止任务堆积
                if self._frame_future is None or self._frame_future.done():
                    # 检查上一 future 是否抛异常
                    if self._frame_future is not None:
                        exc = self._frame_future.exception()
                        if exc is not None:
                            logger.error("previous frame task raised: %s", exc)
                    self._frame_future = self._frame_executor.submit(
                        self._process_frame_bg, frame)

                # 主线程只做显示：cvt + resize + 贴图
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                pil_image = utils.resize_image(pil_image,
                                               self._video_w, self._video_h)
                photo = ImageTk.PhotoImage(image=pil_image)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            else:
                self._camera_fail_count += 1
                if self._camera_fail_count >= _MAX_CAMERA_FAILS:
                    self._reinit_camera()
        except Exception:
            logger.exception("update_video_feed error")
        finally:
            self.root.after(_FRAME_INTERVAL_MS, self._update_video_feed)

    # ================== 关闭清理 ==================
    def _on_close(self):
        # 关闭前保存 GUI 用户配置（尽量提前，避免后续步骤异常时丢失）
        try:
            save_gui_config(
                auto_train=bool(self.auto_train_var.get()),
                camera_index=int(GlobalInfo.camera_index),
                jump_threshold=int(GlobalInfo.gaze_jump_jump_threshold),
                glide_multiplier=float(GlobalInfo.gaze_glide_max_multiplier),
                # 进阶参数
                jump_cooldown_ms=int(GlobalInfo.gaze_jump_cooldown_ms),
                follow_idle_seconds=float(GlobalInfo.gaze_follow_idle_seconds),
                follow_ease=float(GlobalInfo.gaze_follow_ease),
                glide_near_threshold=int(GlobalInfo.gaze_glide_near_threshold),
                glide_far_threshold=int(GlobalInfo.gaze_glide_far_threshold),
                glide_dist_exponent=float(GlobalInfo.gaze_glide_dist_exponent),
            )
        except Exception:
            logger.exception("save gui config failed")

        # 关闭前先停掉视线鼠标模式（否则监听线程会残留）
        try:
            self.cursor_mode_manager.stop_all()
        except Exception:
            logger.exception("stop cursor mode manager failed")
        # 关闭前先隐藏红点悬浮窗，避免残留
        try:
            self.red_dot_overlay.stop()
        except Exception:
            logger.exception("stop red dot overlay failed")

        # 后台线程执行耗时清理，避免阻塞主线程
        def _do_close():
            # 1) 等训练线程收尾（避免写坏模型文件）
            controller = GlobalInfo.train_and_predict_instance
            if controller is not None and getattr(controller, "training", False):
                self.root.after(0, lambda: self.model_state_str.set("等待训练完成后退出…"))
                # 最多等 30 秒；超过就强退
                deadline = time.time() + 30.0
                while getattr(controller, "training", False) and time.time() < deadline:
                    time.sleep(0.2)

            # 2) 保存样本
            try:
                if GlobalInfo.train_data is not None:
                    GlobalInfo.train_data.save_current_sample_to_local()
            except Exception:
                logger.exception("save sample failed")

            # 3) 释放 MediaPipe
            try:
                extractor = getattr(GlobalInfo, 'gaze_feature_extractor', None)
                if extractor is not None:
                    landmarker = getattr(extractor, 'landmarker', None)
                    if landmarker is not None:
                        landmarker.close()
            except Exception:
                logger.exception("landmarker close failed")

            # 4) 释放摄像头
            try:
                if GlobalInfo.video_steam is not None:
                    GlobalInfo.video_steam.release()
            except Exception:
                logger.exception("video release failed")

            # 5) 关闭线程池
            try:
                self._frame_executor.shutdown(wait=False)
            except Exception:
                logger.exception("executor shutdown failed")

            # 6) 主线程销毁窗口
            self.root.after(0, self.root.destroy)

        threading.Thread(target=_do_close, daemon=True, name="CloseThread").start()

    # ================== 启动 ==================
    def run(self):
        # 启动时刷新一次样本累计显示（包含本地已有 + 本次会话）
        try:
            train_data = GlobalInfo.train_data
            if train_data is not None and hasattr(train_data, '_push_sample_count_to_ui'):
                train_data._push_sample_count_to_ui()
        except Exception:
            logger.exception("init sample count display failed")
        # 启动点击监听（保存样本）
        threading.Thread(target=utils.start_listening_click_dnn,
                         daemon=True, name="ClickListener").start()
        # 启动键盘监听（光标随动）
        utils.move_cursor_when_press_key()
        # 启动帧循环
        self._update_video_feed()
        # 初始 mode 状态
        self._on_mode_changed()
        self.root.mainloop()


def run_gui():
    """入口函数：构造 App 并进入 mainloop。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    app = GazeApp()
    app.run()


if __name__ == "__main__":
    run_gui()
