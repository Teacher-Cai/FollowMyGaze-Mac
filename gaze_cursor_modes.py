"""视线-鼠标交互模式：
- GazeJumpController: 手动移动鼠标时，若光标距视线过远则一次性瞬移到视线位置
- GazeFollowController: 光标自动跟随视线，用户操作时让位
- GazeGlideController: 鼠标沿视线方向移动时按 cos + 距离加权加速滑翔

两个 controller 都可 start/stop，通过 GUI 的 mode 切换驱动。
使用 pynput 监听鼠标事件；用 pyautogui 控制鼠标位置。

关键：程序自身的 pyautogui.moveTo 也会触发 on_move 回调。我们用
`_last_programmatic_pos` 记录最近一次程序设置的坐标，并允许一个
`gaze_follow_user_move_pixel` 的容差来判定"用户手动"vs"程序回调"。
"""

import logging
import math
import threading
import time

import pyautogui
from pynput import mouse

from global_info import GlobalInfo

# 关掉 pyautogui 的 fail-safe（鼠标到屏幕角就抛异常）——本模块自己做边界保护。
# 这个模块被 import 时就设一次即可（全局生效）。
pyautogui.FAILSAFE = False
# pyautogui 默认每次 moveTo 会 sleep 0.1s，串起来会明显延迟，禁掉
pyautogui.PAUSE = 0

logger = logging.getLogger(__name__)


def _gaze_xy():
    """当前视线预测点（可能为 0，需要调用方判断）"""
    return GlobalInfo.red_dot_x, GlobalInfo.red_dot_y


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# =============================================================
# Mode: gaze_jump — 距离过远时手动移动鼠标 → 瞬移到视线位置
# =============================================================
class GazeJumpController:
    """监听鼠标手动移动。当满足以下条件时，把光标瞬移到视线预测点：

    - 用户手动移动了鼠标（单次移动 >= min_user_move 像素）
    - 当前光标位置与视线预测点的距离 > jump_threshold
    - 距上一次瞬移已过了 cooldown_ms（防止视线抖动导致来回跳）

    效果：
    - 手轻推一下鼠标 → 直接跳到目光所在处
    - 到达后视线点附近的精细定位交给用户继续手动移动
    """

    def __init__(self):
        self._listener = None
        self._last_pos = None                 # 上次观察到的鼠标位置
        self._suppress_next = False           # 是否忽略下一次 on_move（我们刚 moveTo 过）
        self._last_jump_time = 0.0            # 上次瞬移时间戳（秒）
        self._lock = threading.Lock()

    def start(self):
        if self._listener is not None:
            return
        self._last_pos = pyautogui.position()
        self._suppress_next = False
        self._last_jump_time = 0.0
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.daemon = True
        self._listener.start()
        logger.info("GazeJumpController started")

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop jump listener failed")
            self._listener = None
        logger.info("GazeJumpController stopped")

    def _on_move(self, x, y):
        # 防御 pynput 偶发的 inf/nan 坐标（macOS 屏幕边缘 / 多屏切换时出现过）
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        with self._lock:
            if self._last_pos is None:
                self._last_pos = (x, y)
                return

            # 过滤：这是我们自己 moveTo 触发的回调
            if self._suppress_next:
                self._suppress_next = False
                self._last_pos = (x, y)
                return

            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if not (math.isfinite(dx) and math.isfinite(dy)):
                self._last_pos = (x, y)
                return

            # 用户是否在明显地移动？太小的抖动忽略
            move_mag = math.hypot(dx, dy)
            if move_mag < GlobalInfo.gaze_jump_min_user_move:
                self._last_pos = (x, y)
                return

            # 冷却期内不再瞬移，等用户先"消化"上一次跳跃
            now = time.time()
            cooldown = GlobalInfo.gaze_jump_cooldown_ms / 1000.0
            if now - self._last_jump_time < cooldown:
                self._last_pos = (x, y)
                return

            # 需要有视线预测
            gx, gy = _gaze_xy()

            # 光标距视线距离 → 超阈值才瞬移
            distance = _dist(x, y, gx, gy)
            if not math.isfinite(distance):
                self._last_pos = (x, y)
                return
            if distance < GlobalInfo.gaze_jump_jump_threshold:
                self._last_pos = (x, y)
                return

            # 触发瞬移：跳到视线位置（做边界保护）
            new_x, new_y = gx, gy
            self._suppress_next = True
            self._last_jump_time = now
            try:
                pyautogui.moveTo(new_x, new_y, _pause=False)
                logger.debug("gaze jump: (%d,%d) -> gaze (%d,%d), dist=%.1f",
                             int(x), int(y), new_x, new_y, distance)
            except Exception:
                logger.exception("jump moveTo failed")
            self._last_pos = (new_x, new_y)


# =============================================================
# Mode: gaze_follow — 光标自动跟随视线，用户操作时让位
# =============================================================
class GazeFollowController:
    """后台线程持续把光标移向视线点；检测到用户操作 → 暂停 idle_seconds。

    状态机：
      FOLLOWING   → 用户 on_move (delta > threshold) → PAUSED (记录 last_user_time)
      PAUSED      → 距 last_user_time > idle_seconds → FOLLOWING

    区分程序 vs 用户移动：用 _last_programmatic_pos 比对；
    on_move 的坐标如果和 _last_programmatic_pos 距离 <= user_move_pixel 视为程序移动。
    """

    def __init__(self):
        self._listener = None
        self._thread = None
        self._stop_event = threading.Event()

        self._paused = False
        self._last_user_time = 0.0
        self._last_programmatic_pos = None   # 最近一次自动跟随设置的目标
        self._last_seen_pos = None           # 上一帧观察到的光标位置（用于检测用户是否还在动）
        self._lock = threading.Lock()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._paused = False
        self._last_user_time = 0.0
        self._last_programmatic_pos = None
        self._last_seen_pos = None

        # 只监听点击/滚动作为"用户操作"信号；移动检测改到 follow_loop 里轮询，
        # 避免 pynput 坐标(物理像素)与 pyautogui(逻辑像素)的缩放不一致问题。
        self._listener = mouse.Listener(
            on_click=self._on_activity, on_scroll=self._on_activity)
        self._listener.daemon = True
        self._listener.start()

        self._thread = threading.Thread(
            target=self._follow_loop, daemon=True, name="GazeFollow")
        self._thread.start()
        logger.info("GazeFollowController started")

    def stop(self):
        self._stop_event.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop follow listener failed")
            self._listener = None
        # 后台线程 daemon，不需要 join
        logger.info("GazeFollowController stopped")

    def _on_move(self, x, y):
        # 已弃用：移动检测移到 _tick 轮询里。保留空实现以防未来接回。
        return

    def _on_activity(self, *args, **kwargs):
        with self._lock:
            self._paused = True
            self._last_user_time = time.time()

    def _follow_loop(self):
        interval = max(0.005, GlobalInfo.gaze_follow_step_interval_ms / 1000.0)
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("gaze follow tick failed")
            time.sleep(interval)

    def _tick(self):
        cur_x, cur_y = pyautogui.position()

        # —— 用户移动检测（每帧都做，暂停期间也不停）——
        # 核心思路：程序的 moveTo 是精确的，上一帧我们把光标移到了 _last_programmatic_pos。
        # 这一帧读到的光标位置只要偏离那个"预期位置"超过一个小容差，就一定是用户手动移动的
        # （无论快慢）。这样即使慢速精细移动，也能被立刻识别，不会被程序抢走。
        tol = GlobalInfo.gaze_follow_user_move_pixel
        with self._lock:
            prog = self._last_programmatic_pos

            user_moving = False
            if prog is not None:
                drift = _dist(cur_x, cur_y, prog[0], prog[1])
                if math.isfinite(drift) and drift > tol:
                    user_moving = True
            # prog 为 None 时（尚未跟随过 / 刚让位）不主动判定移动，靠下方 seen 兜底

            self._last_seen_pos = (cur_x, cur_y)

            # 只要用户还在动，就持续刷新计时起点 → 停下来后才开始倒计时
            if user_moving:
                self._paused = True
                self._last_user_time = time.time()

            paused = self._paused
            idle_elapsed = time.time() - self._last_user_time

        if paused:
            if idle_elapsed < GlobalInfo.gaze_follow_idle_seconds:
                # 让位期间：把"程序预期位置"锁定为当前真实光标位置，
                # 这样用户继续移动时，下一帧 drift 依然能检测到（持续刷新计时）。
                with self._lock:
                    self._last_programmatic_pos = (cur_x, cur_y)
                return
            # 用户已彻底停手超过 idle_seconds → 恢复跟随
            with self._lock:
                self._paused = False

        gx, gy = _gaze_xy()
        # 没有有效视线预测 → 不动
        if not (math.isfinite(gx) and math.isfinite(gy)) or (gx == 0 and gy == 0):
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
            return

        # 缓动移动：每步只走一部分距离，避免瞬跳
        ease = max(0.0, min(1.0, GlobalInfo.gaze_follow_ease))
        nx = int(round(cur_x + (gx - cur_x) * ease))
        ny = int(round(cur_y + (gy - cur_y) * ease))

        # 太小的移动跳过（但仍把预期位置锁到当前光标，保证下一帧能检测用户移动）
        if abs(nx - cur_x) < 1 and abs(ny - cur_y) < 1:
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
                self._last_seen_pos = (cur_x, cur_y)
            return

        try:
            pyautogui.moveTo(nx, ny, _pause=False)
        except Exception:
            logger.exception("follow moveTo failed")
            with self._lock:
                self._last_programmatic_pos = (cur_x, cur_y)
            return
        # moveTo 之后记录"程序期望光标所在位置"，供下一帧对比
        with self._lock:
            self._last_programmatic_pos = (nx, ny)
            self._last_seen_pos = (nx, ny)


# =============================================================
# Mode: gaze_glide — 鼠标沿视线方向移动时加速滑翔
# =============================================================
class GazeGlideController:
    """当用户移动鼠标的方向与"当前光标位置→视线点"的方向一致时，放大位移。

    关键设计（修复"越过后被拉回"的 bug）：
      gaze_dir 从"当前光标位置"指向视线点，而不是从 stroke 起点指向视线点。
      这样一旦光标越过视线点，gaze_dir 立刻翻转，cos 变负，加速自动停止；
      甚至用户继续朝原方向推 → cos < 0 → dir_factor = 0，绝不再加速冲远。

    速度系数 factor = 1 + (max_mul - 1) * dist_factor * dir_factor
      - dist_factor: 距视线距离越远越接近 1；<= near_threshold 时为 0
      - dir_factor:  cos(user_dir, gaze_dir) 归一化到 [0,1]

    自身回声过滤：用最近一次程序 moveTo 的目标位置 + 容差比对，
    而非只拦截"下一次" on_move（macOS 一次 moveTo 会产生多次回调）。
    """

    def __init__(self):
        self._listener = None
        self._last_pos = None            # 上次观察到的鼠标位置
        self._last_move_time = 0.0       # 上次移动时间戳（未来若需 stroke 逻辑保留）
        # 自身回声过滤：记录最近程序设置的目标位置，若 on_move 坐标离它足够近就丢弃
        self._last_programmatic = None
        self._lock = threading.Lock()

    def start(self):
        if self._listener is not None:
            return
        pos = pyautogui.position()
        self._last_pos = pos
        self._last_move_time = 0.0
        self._last_programmatic = None
        self._listener = mouse.Listener(on_move=self._on_move)
        self._listener.daemon = True
        self._listener.start()
        logger.info("GazeGlideController started")

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("stop glide listener failed")
            self._listener = None
        logger.info("GazeGlideController stopped")

    # ---- 系数计算 ----
    def _dist_factor(self, distance):
        """距离因子：距目标越近越接近 0（柔和刹车），越远越接近 1。

        使用指数曲线 (linear ** exponent) 让近处衰减更快，避免用户手推冲过目标：
          exponent=1  → 线性（旧行为）
          exponent=2  → 平方衰减，距 near 1/2 处 df≈0.25
          exponent=3  → 立方衰减，更保守
        """
        near = GlobalInfo.gaze_glide_near_threshold
        far = GlobalInfo.gaze_glide_far_threshold
        if distance <= near:
            return 0.0
        if distance >= far:
            return 1.0
        linear = (distance - near) / max(1e-6, (far - near))
        exponent = getattr(GlobalInfo, "gaze_glide_dist_exponent", 1.0)
        try:
            exponent = float(exponent)
        except Exception:
            exponent = 1.0
        if exponent <= 0:
            exponent = 1.0
        # linear ∈ [0,1]，pow 保号
        return max(0.0, min(1.0, linear ** exponent))

    def _dir_factor(self, mouse_dir, gaze_dir):
        """cos 加权，返回 [0, 1]。cos <= cos_threshold → 0；cos = 1 → 1。"""
        mx, my = mouse_dir
        gx, gy = gaze_dir
        mn = math.hypot(mx, my)
        gn = math.hypot(gx, gy)
        if mn <= 1e-6 or gn <= 1e-6:
            return 0.0
        cos_v = (mx * gx + my * gy) / (mn * gn)
        cos_v = max(-1.0, min(1.0, cos_v))
        cos_th = GlobalInfo.gaze_glide_cos_threshold
        if cos_v <= cos_th:
            return 0.0
        return cos_v

    def _on_move(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        with self._lock:
            now = time.time()

            # ★ 关键修复：使用 pyautogui.position() 读取"实时真实光标位置"，
            # 而不是 pynput 事件里滞后的 (x, y)。
            # 原因：当用户手速远快于事件队列处理速度时，事件坐标 (x, y) 是几十毫秒前的
            # 旧位置；如果我们基于旧位置 moveTo(new_x)（绝对坐标），
            # 会把已经跑到远处的真实光标"硬拽回"旧位置附近，产生"划走了又被拉回"的现象。
            try:
                real_x, real_y = pyautogui.position()
            except Exception:
                real_x, real_y = x, y
            if not (math.isfinite(real_x) and math.isfinite(real_y)):
                real_x, real_y = x, y

            if self._last_pos is None:
                self._last_pos = (real_x, real_y)
                self._last_move_time = now
                return

            # 自身回声过滤：本次事件坐标 ≈ 最近一次程序 moveTo 目标 → 丢弃
            prog = self._last_programmatic
            if prog is not None:
                if _dist(x, y, prog[0], prog[1]) <= max(2, GlobalInfo.gaze_glide_min_stroke_len):
                    self._last_pos = (real_x, real_y)
                    self._last_move_time = now
                    self._last_programmatic = None
                    return

            # 用户本次移动方向：用事件坐标增量（这才反映"物理手推的方向"）
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if not (math.isfinite(dx) and math.isfinite(dy)):
                self._last_pos = (real_x, real_y)
                self._last_move_time = now
                return
            if dx == 0 and dy == 0:
                return
            self._last_move_time = now

            gx, gy = _gaze_xy()

            # 距离因子：用"真实光标位置"和视线的距离
            distance = _dist(real_x, real_y, gx, gy)
            if not math.isfinite(distance):
                self._last_pos = (real_x, real_y)
                return
            df = self._dist_factor(distance)
            if df <= 0.0:
                self._last_pos = (real_x, real_y)
                return

            # 方向因子：mouse_dir 用事件增量；gaze_dir 用"真实光标 → 视线"
            mouse_dir = (dx, dy)
            gaze_dir = (gx - real_x, gy - real_y)
            if math.hypot(dx, dy) < max(1, GlobalInfo.gaze_glide_min_stroke_len):
                self._last_pos = (real_x, real_y)
                return
            rf = self._dir_factor(mouse_dir, gaze_dir)
            if rf <= 0.0:
                self._last_pos = (real_x, real_y)
                return

            max_mul = GlobalInfo.gaze_glide_max_multiplier
            factor = 1.0 + (max_mul - 1.0) * df * rf
            if not math.isfinite(factor) or factor <= 1.001:
                self._last_pos = (real_x, real_y)
                return

            # 额外增量：只补 (factor - 1) 倍的位移，不覆盖用户已产生的物理位移
            extra_dx_f = dx * (factor - 1.0)
            extra_dy_f = dy * (factor - 1.0)
            if not (math.isfinite(extra_dx_f) and math.isfinite(extra_dy_f)):
                self._last_pos = (real_x, real_y)
                return

            # 越界保护（柔和刹车）：额外位移不超过"当前到 near 圈的剩余距离"，
            # 保证即便用户还在快推，光标也不会冲进 near_threshold 圈内。
            # 相当于给光标画了一个"以视线点为圆心、半径 = near * anchor_ratio 的软停车带"。
            near = GlobalInfo.gaze_glide_near_threshold
            anchor_ratio = getattr(GlobalInfo, "gaze_glide_overshoot_anchor_ratio", 1.0)
            try:
                anchor_ratio = float(anchor_ratio)
            except Exception:
                anchor_ratio = 1.0
            near_anchor = max(0.0, near * anchor_ratio)
            remaining = max(0.0, distance - near_anchor)
            extra_mag = math.hypot(extra_dx_f, extra_dy_f)
            if extra_mag > remaining:
                scale = remaining / max(1e-6, extra_mag)
                extra_dx_f *= scale
                extra_dy_f *= scale

            extra_dx = int(round(extra_dx_f))
            extra_dy = int(round(extra_dy_f))
            if extra_dx == 0 and extra_dy == 0:
                self._last_pos = (real_x, real_y)
                return

            # ★ 关键修复：改用相对位移 pyautogui.move(dx, dy)
            # 相对位移不会"拉回"用户已经推走的光标，只会在真实位置上叠加增量。
            # 这样即使 pynput 事件延迟了 100ms，用户手已经把光标从 (100,100) 推到 (900,100)，
            # 我们的 extra_dx=+50 会作用在真实位置 (900,100) 上 → 落到 (950,100)，
            # 而不是被 moveTo(150,100) 硬拉回 (150,100)。
            try:
                pyautogui.move(extra_dx, extra_dy, _pause=False)
                # 记录调用后估计的目标点（用于回声过滤）
                self._last_programmatic = (real_x + extra_dx, real_y + extra_dy)
                logger.debug("glide: factor=%.2f df=%.2f rf=%.2f dist=%.1f extra=(%d,%d)",
                             factor, df, rf, distance, extra_dx, extra_dy)
            except Exception:
                logger.exception("glide move failed")

            # last_pos 用事件坐标（下次算 dx/dy 才连贯）
            self._last_pos = (x, y)


# =============================================================
# 统一管理器：按 mode 名字启停
# =============================================================
class CursorModeManager:
    """根据 mode 名字启停对应 controller，确保任何时刻只有一个激活。"""

    def __init__(self):
        self._jump = GazeJumpController()
        self._follow = GazeFollowController()
        self._glide = GazeGlideController()
        self._current = None    # 'gaze_jump' | 'gaze_follow' | 'gaze_glide' | None

    def switch_to(self, mode_name):
        if mode_name == self._current:
            return
        # 关闭当前
        if self._current == 'gaze_jump':
            self._jump.stop()
        elif self._current == 'gaze_follow':
            self._follow.stop()
        elif self._current == 'gaze_glide':
            self._glide.stop()
        # 开启新的
        if mode_name == 'gaze_jump':
            self._jump.start()
        elif mode_name == 'gaze_follow':
            self._follow.start()
        elif mode_name == 'gaze_glide':
            self._glide.start()
        self._current = mode_name if mode_name in (
            'gaze_jump', 'gaze_follow', 'gaze_glide') else None

    def stop_all(self):
        self.switch_to(None)
