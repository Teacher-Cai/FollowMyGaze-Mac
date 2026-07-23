import os


class GlobalInfo:
    video_steam = None
    screen_width = None
    screen_height = None
    train_and_predict_instance = None
    model_state_var = None
    sample_count_var = None  # tkinter StringVar，用于实时显示累计样本数
    predict_executor = None  # ThreadPoolExecutor，用于复用推理线程，避免每帧创建新线程
    sample_upper_limit = 100000  # 10w
    train_data = None
    current_frame = None
    gaze_feature_extractor = None
    current_features = None
    online_training_batchSize = 2048
    online_training_epoch = 40

    offline_training_batchSize = 2048
    offline_training_epoch = 400

    # —— 在线自动训练配置 ——
    # 每累积 auto_train_threshold 个新样本，自动触发一次“全量样本”训练
    # 训练在后台线程执行，不阻塞采集；epoch 数复用 online_training_epoch
    auto_train_threshold = 1024
    enable_auto_train = True

    red_dot_x = 0
    red_dot_y = 0
    mode_select = None
    enable_move_cursor = False
    move_cursor_x = 0
    move_cursor_y = 0
    show_red_dot_win = False
    # 红点显示独立开关（与 mode 解耦）：勾选就显示红点，无论当前处于 silent_train / move_cursor
    show_red_dot_enabled = False

    # ================== 新增鼠标交互模式配置 ==================
    # —— 模式 gaze_jump：手动移动鼠标时，若光标距视线过远则一次性瞬移到视线位置 ——
    # 触发条件：用户正在手动移动鼠标，且当前光标位置与视线预测点距离 > jump_threshold
    # 瞬移后进入 cooldown 期，避免视线抖动导致连续瞬移
    gaze_jump_jump_threshold = 300       # 触发瞬移的最小距离（像素）
    gaze_jump_cooldown_ms = 2000          # 瞬移冷却时间（毫秒）
    gaze_jump_min_user_move = 5          # 用户单次移动 >= 此像素才判定为"正在操作"，避免误触发

    # —— 模式 gaze_follow：光标自动跟随视线，用户操作时让位 ——
    gaze_follow_idle_seconds = 3.0        # 用户停止操作后 N 秒恢复自动跟随
    gaze_follow_user_move_pixel = 3       # 光标偏离"程序预期位置" > 此值 → 判定为用户操作（越小越灵敏）
    gaze_follow_step_interval_ms = 30     # 自动跟随的步进间隔
    gaze_follow_ease = 0.35               # 每步向目标移动的比例（0~1，越小越顺滑）

    # —— 模式 gaze_glide：鼠标移动方向与视线方向一致时加速滑翔 ——
    # 距离越远（远于 near_threshold）越接近满速；越近越回归正常
    # 方向用 cos 加权：cos = 1 满加速，cos <= cos_threshold 不加速；中间线性插值
    gaze_glide_max_multiplier = 5.0       # 最大加速倍数（滑块可调）
    gaze_glide_near_threshold = 300       # 距视线 <= 该值 → 不加速（正常速度）
    gaze_glide_far_threshold = 500        # 距视线 >= 该值 → 距离因子取最大
    gaze_glide_cos_threshold = 0.6        # cos <= 该值视为方向不一致，不加速
    gaze_glide_stroke_reset_ms = 200      # 鼠标停顿超过此时长 → 重置 stroke 起点
    gaze_glide_min_stroke_len = 20         # stroke 长度 < 此像素时暂不启用方向判定（避免抖动噪声）
    # —— 柔和刹车相关 ——
    # 距离因子曲线指数：>1 时近处衰减更快（"接近目标就大幅减速"）；=1 退化为原线性
    gaze_glide_dist_exponent = 2.0
    # 越界锚：extra 位移最多让光标落到"距目标 near_threshold * anchor_ratio"处
    # anchor_ratio = 1.0 → 严格不进入 near 圈；<1.0 → 允许进入一点；>1.0 → 更保守
    gaze_glide_overshoot_anchor_ratio = 0.5

    root = None
    # 用户数据目录：打包后 .app 工作目录是 /，必须用绝对路径
    path_dir = os.path.join(os.path.expanduser("~"), "FollowMyGaze")
