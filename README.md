# FollowMyGaze

> 用眼睛替手腕省下每天上千次的鼠标位移，降低腱鞘炎风险、提升工作效率。
> 最终目标：让视线成为你操作电脑的主力，逐步替代鼠标。

<p>
  <img alt="platform" src="https://img.shields.io/badge/platform-macOS-lightgrey">
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-red">
  <img alt="mediapipe" src="https://img.shields.io/badge/MediaPipe-FaceLandmarker-green">
  <img alt="status" src="https://img.shields.io/badge/status-experimental-orange">
</p>

FollowMyGaze 是一个跑在个人电脑上的**个性化视线追踪 + 视线-鼠标混合交互**工具。
它以普通摄像头作为输入，用 [MediaPipe FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) 提取人脸/眼部几何特征，训练一个基于 **Mixture-of-Experts + 物理先验门控** 的 PyTorch 小模型，把预测到的视线坐标用于红点显示、光标随动、视线跳转、视线跟随和视线滑翔等多种交互模式。

## 项目定位

**本项目的核心目标是减少鼠标的操作。** 现代办公场景中，鼠标点击与拖动是极度高频、极度重复的动作，长期累积会带来腕管综合征、腱鞘炎等劳损风险，也拖慢了操作效率。FollowMyGaze 希望：

- **减少鼠标操作** —— 把最累的大范围光标位移从手腕转移到眼睛。
- **降低腱鞘炎 / 鼠标手风险** —— 减少手腕、手指的重复微动作和静态负荷。
- **提高工作效率** —— 视线擅长快速大范围转移，手只负责最后一厘米的精确定位。
- **最终替代鼠标** —— 长期目标是让视线成为操作电脑的主力输入方式。

**一个额外的副作用是：它同时是一个轻量的深度学习实验平台。** 因为"模型好不好，眼睛立刻能看出来"，从数据采集、特征工程、模型训练、shadow-model 部署到实时推理与交互层一应俱全，非常适合"改一行代码、立即用眼睛验证"的快速实验。

整个系统的核心特点：

- **强个性化**：你自己每次点击都会成为一条训练样本，模型持续在线学习你的头姿+眼动习惯。
- **强可解释性**：手工设计的 124 维几何特征 + 显式拆分头姿/虹膜两个专家，改哪儿动了哪儿一目了然。
- **强工程完备性**：从数据采集、特征工程、模型训练、shadow-model 部署、实时推理、到鼠标交互层，麻雀虽小五脏俱全。

> ⚠️ 当前项目定位是个人实验/原型工具，不是通用 SDK。使用需摄像头、辅助功能、输入监控三个权限；模型效果强依赖于个人采样质量。

---

## 目录

- [项目定位](#项目定位)
- [项目背景](#项目背景)
  - [视角一：降低鼠标手 / 腱鞘炎风险](#视角一给长时间用电脑的人降低鼠标手--腱鞘炎风险)
  - [视角二：轻量的深度学习实验平台](#视角二作为轻量的深度学习实验平台)
- [功能特性](#功能特性)
- [快速上手](#快速上手)
  - [环境要求](#环境要求)
  - [启动](#启动)
  - [基本使用流程](#基本使用流程)
- [交互模式](#交互模式)
- [模型设计思路](#模型设计思路)
- [训练与推理机制](#训练与推理机制)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [打包](#打包)
- [开发建议](#开发建议)
- [未来展望](#未来展望)
- [License](#license)

---

## 项目背景

FollowMyGaze 的定位不是"再造一个通用视线追踪 SDK"，而是同时服务于两类看起来不相关、但都受益于**"视线驱动 + 个性化学习"**的场景。理解这两个视角，能帮助你决定要不要把它跑起来、以及以什么姿势跑起来。

### 视角一：给长时间用电脑的人，降低鼠标手 / 腱鞘炎风险

现代办公场景中，鼠标操作已经是**极度高频、极度重复**的动作：

- 每天数千次点击
- 大量小幅度腕部微调（把光标从 A 拖到 B）
- 长时间保持"前臂前伸 + 手腕悬空 + 拇指外展"的静态姿势

这些动作长期累积会引发几种常见问题：

- **腕管综合征**（Carpal Tunnel Syndrome）：正中神经受压，出现手指麻木、握力下降
- **腱鞘炎**（De Quervain / trigger finger）：反复摩擦让肌腱鞘发炎，拇指、食指、腕部酸痛
- **鼠标肘 / 网球肘**：外上髁反复受力发炎
- **肩颈紧张**：长期悬肘操作让斜方肌和肩胛提肌持续紧绷

**FollowMyGaze 的价值不在于"取代鼠标"，而在于把每一次光标位移中最累的部分——大范围移动——从手腕转移到眼睛上。** 具体机制：

1. **消除大部分长距离拖动**
   看向哪儿，光标就跳/滑到哪儿。屏幕上从左上到右下的大跨度移动，原本需要抬腕 + 前臂配合完成，现在几乎不需要用手就能完成 90% 以上的距离。

2. **手只做"最后一厘米"精细定位**
   `视线滑翔` / `视线跳转` 模式都是**远距离视线接管、近距离交给手**。这正好利用了两者的优势：视线擅长快速大范围转移，手擅长小范围精确定位。

3. **减少手指/手腕微动作次数**
   同样是从代码编辑器切到浏览器点某个按钮，传统方式是"移动鼠标 → 点击"，可能需要 10+ 次小幅调整；用视线滑翔后，一次快速手推 + 视线聚焦即可到达目标附近。

4. **降低静态负荷时间**
   传统鼠标操作要求手时刻"待命"在鼠标上；FollowMyGaze 的多种模式允许你**手完全离开鼠标**去做别的事（阅读、思考），只在真需要点击时再触碰。

5. **保留传统鼠标的可控性**
   项目故意没做成"完全脱手视线控制"，而是提供**多种混合模式**（Alt 触发、视线跳转触发、视线滑翔加速）——你的手仍然处于主控地位，视线只做辅助放大器。这样既减负，又不牺牲精度和可预测性。

> 这不是医疗器械声明，仅是从操作学和人体工学角度的合理推断。真正有 RSI（重复性劳损）相关症状请就医。

**适合这些人群试用**：

- 每天键鼠时间超过 6 小时的程序员、设计师、运营
- 已经有轻微腕部不适、想主动降低使用强度的人
- 使用大屏幕 / 双显示器、鼠标跨屏距离远的用户
- 希望在阅读、看视频时减少手部占用的用户

### 视角二：作为轻量的深度学习实验平台

如果说健康层面的价值面向**每一位重度用户**，那么工程层面的价值就是面向**每一位对深度学习感兴趣的开发者**。

绝大多数深度学习入门项目停留在"训练一个 MNIST / CIFAR 分类器"，你能看到的只是准确率数字的变化。FollowMyGaze 提供了一个截然不同的学习闭环：**模型好不好，眼睛立刻能看出来**。

这带来了几个独特的教学 / 实验优势：

1. **即时的、可视的反馈**
   模型改一版、训练一次，打开红点追踪窗口，直接用眼睛判断：红点跟你视线的偏差有多大？在屏幕四角是否失灵？是否有明显抖动或漂移？——这比看 loss 曲线直观得多，也更容易触发直觉性思考。

2. **完整覆盖深度学习工程链路**
   项目里刚好包含了工程上"从数据到部署"的所有关键环节，且每一环都足够小、可读、可改：
   - **数据采集**：`utils.py` 的鼠标点击监听
   - **特征工程**：`gaze_feature_extractor_mac.py` 的手工几何特征（124 维）
   - **模型结构**：`gaze_feature_based_model.py` 的 MoE + CrossNet + ResBlock
   - **损失设计**：`train_and_predict_dnn.py` 的多任务加权损失
   - **训练循环**：Adam + BN + Dropout，全量/在线两套配置
   - **在线学习**：`data_process_dnn.py` 的自动触发训练
   - **模型部署**：Shadow model + 原子替换 + EMA 平滑
   - **实时推理**：Tk 主线程 + 后台线程池 + 摄像头 30fps
   - **交互层**：多种鼠标控制模式验证模型效果

3. **每个改动都能立即验证**
   - 加一维新特征 → 训练 → 看红点是否更稳
   - 改门控阈值策略 → 训练 → 看头姿变化时的漂移
   - 换个损失权重 → 训练 → 看不同眼动状态下的误差分布
   - 调 EMA alpha → 感受平滑度和延迟的权衡

4. **数据规模适中，实验成本低**
   本地采集几千个样本，训练一次几十秒到几分钟。既不像 toy 数据集那样脱离真实工程，又不像大型项目那样一个实验要等一天。**"改代码 → 训练 → 用眼睛验证"整个循环控制在 5 分钟内**，非常适合快速迭代。

5. **暴露真实工程问题**
   这不是一个纯净的算法练习题，而是一个**必须解决工程细节**的完整系统：
   - 训练线程和推理线程如何共享模型？（→ Shadow model 模式）
   - BatchNorm 在推理时会不会污染 running stats？（→ eval/train 隔离）
   - 摄像头掉帧怎么办？（→ 自动重连 + 帧任务防堆积）
   - 事件流延迟导致鼠标被拉回怎么办？（→ 相对位移 vs 绝对坐标）
   - 训练时 UI 卡死怎么办？（→ 全部后台线程 + `root.after`）

   这些坑几乎是所有实时 AI 系统都会遇到的，值得亲手踩一遍。

6. **鼓励物理直觉驱动的模型设计**
   注视点 = 头姿贡献 + 虹膜偏转贡献。这个物理直觉直接决定了模型是 MoE 结构，门控用 `||rel||` 物理先验来路由。这种"从领域先验推导网络结构"的思路，是从 toy 例子里学不到的，但正是工业界模型设计的核心方法论。

**适合这样使用**：

- 深度学习新手：把 `SimpleDNN` 和 `GazeMoE` 拿出来对比，理解为什么加 residual、加 BN、加 gate
- 中级开发者：试试改特征、改结构、改损失，训练 → 用眼睛验证
- 想学习"实时 AI 系统工程"的人：整个 `gui.py` + `train_and_predict_dnn.py` 就是一个小型 online-learning 平台的完整实现

---

## 功能特性

围绕上面两个视角，项目提供以下能力：

- 🎥 **实时摄像头预览**：Tkinter GUI 显示摄像头画面。
- 🧠 **MediaPipe 人脸特征提取**：通过 `face_landmarker.task` 提取人脸、虹膜、头部姿态相关特征。
- 🖱️ **点击采样**：左键点击屏幕时，将当前视线特征与点击坐标保存为训练样本。
- 💾 **样本持久化**：样本保存到本地 `~/FollowMyGaze/samples/samples.pkl`。
- 🔢 **样本计数显示**：显示 `本地样本 + 本次会话样本 = 总样本数`。
- 🏋️ **手动训练**：一键"用全部数据训练"，后台跑全量样本。
- ⚙️ **自动训练**：每累计 `auto_train_threshold` 个新样本后自动持久化并后台全量训练。
- 🌗 **影子模型训练**：训练使用 shadow model，完成后原子替换推理模型，训练/推理线程互不干扰。
- 🔴 **红点追踪窗口**：显示当前预测视线位置，方便肉眼校验模型效果。
- ✋ **多种鼠标交互模式**：后台训练 / Alt 光标随动 / 视线跳转 / 视线跟随 / 视线滑翔。
- 🎛️ **参数持久化**：GUI 各交互参数在退出时保存到 `user_config.py` 指定的配置文件，下次启动自动恢复。


---

## 快速上手

> **TL;DR（3 步跑起来）**
> ```bash
> pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput appdirs
> python main.py
> ```
> 然后在弹出的授权提示里允许 **摄像头 / 辅助功能 / 输入监控** 三个权限 → 看向屏幕点几下左键采样 → 点"用全部数据训练" → 打开红点追踪看效果。

### 环境要求

推荐环境：

- macOS（其他平台需自行调整特征提取与鼠标控制部分）
- Python 3.9+
- 摄像头
- 已安装 `face_landmarker.task`（仓库已附带）
- 需要授予以下 macOS 隐私权限（**三者缺一不可**）：
  - **摄像头**（Camera）：采集视频用于人脸/视线特征提取
  - **辅助功能**（Accessibility）：`pyautogui` **控制/移动**鼠标（红点、视线跟随等模式）
  - **输入监控**（Input Monitoring）：`pynput` **全局监听**鼠标点击以采集训练样本
  > ⚠️ **辅助功能 与 输入监控 是两个不同的权限**，在系统设置里是分开的两栏，容易混淆：
  > 「辅助功能」负责**控制**鼠标，「输入监控」负责**监听**点击。**缺少输入监控时，会出现"应用在前台能采集、切到后台就不采集"的现象**（未授权时全局事件监听被系统降级为仅接收前台事件）。
  >
  > 从终端 / IDE 运行 `python main.py` 时，通常继承了终端/IDE 已有的权限，所以不易察觉；**打包成 `.app` 后是独立身份，必须单独逐一授权**。
  > 另外：**ad-hoc 签名的 `.app` 每次重新打包，签名指纹（cdhash）都会变化，导致已授予的权限失效，需要重新授权**。

主要 Python 依赖：

```bash
pip install opencv-python pyautogui mediapipe numpy pandas torch pillow pynput appdirs
```

Apple Silicon 用户建议按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装匹配自己环境的 PyTorch 版本。

### 启动

在项目根目录执行：

```bash
python main.py
```

启动后会出现主窗口：

- **上方**：摄像头画面
- **中部**：提示信息、样本计数、当前模式
- **模式选择**：后台训练 / 光标随动(Alt) / 视线跳转 / 视线跟随 / 视线滑翔
- **控制区**：红点追踪按钮、自动训练开关，以及按模式分组的参数面板：
  - **视线跳转**：触发距离、冷却时间
  - **视线跟随**：操作后暂停、跟随顺滑度
  - **视线滑翔**：加速倍数、减速起始距离、全速加速距离、减速陡峭度
- **底部**：手动训练按钮和训练进度条

> 参数面板的取值会在退出时自动持久化（`user_config.py`），下次启动自动恢复。

### 基本使用流程

**1. 启动程序**

```bash
python main.py
```

确保摄像头画面正常显示。

**2. 采集样本**

保持头部和眼睛自然状态，看向屏幕上的某个位置，然后用鼠标左键点击该位置。

每次左键点击会保存：

- 当前帧提取到的 gaze 特征
- 鼠标点击坐标 `(x, y)`

样本会先进入本次会话缓存，并在以下时机持久化：

- 每达到 `GlobalInfo.auto_train_threshold` 个新样本时自动保存
- 程序退出时保存未持久化样本

样本文件路径：

```text
~/FollowMyGaze/samples/samples.pkl
```

**3. 训练模型**

方式一：手动训练。点击 GUI 中的 "用全部数据训练"，程序会加载本地所有样本训练模型，并显示训练进度。

方式二：自动训练。默认开启：

```python
GlobalInfo.enable_auto_train = True
GlobalInfo.auto_train_threshold = 1024
```

每累计 1024 个新样本会触发一次后台全量训练，使用 `online_training_epoch` 作为 epoch 数。训练完成后的模型保存到：

```text
~/FollowMyGaze/gaze_model_resnet.pth
```

下次启动时会自动加载该模型。

**4. 查看预测效果**

点击 "显示红点追踪" 会打开一个覆盖屏幕的红点窗口，红点位置由当前模型预测的视线坐标驱动。

关闭方式：`Esc` / 点击红点窗口 / 关闭窗口。

**建议的第一次使用节奏**：

1. 采集 200~500 个样本，覆盖屏幕四角和中央
2. 用 "用全部数据训练" 手动训一次
3. 打开红点追踪窗口感受模型效果
4. 效果不理想就继续采样 + 训练；效果 OK 就切换交互模式使用

---

## 交互模式

模式切换由 `gaze_cursor_modes.py::CursorModeManager` 统一管理。所有模式都能与红点追踪并存显示。

### 1. 后台训练

GUI 默认模式。只采样和预测，不主动移动鼠标。适合：

- 采集训练数据
- 观察红点预测效果
- 安全调试

### 2. 光标随动(Alt)

- 模型持续预测视线坐标
- 按住 `Alt` 键时，鼠标会循环移动到预测视线位置
- 松开 `Alt` 后停止移动

适合低频触发式控制，不会一直抢鼠标。

### 3. 视线跳转（`gaze_jump`）

**触发逻辑**：用户手动移动鼠标时，如果当前鼠标位置与视线预测点距离超过阈值，鼠标会**一次性跳转**到视线位置附近。跳转后进入冷却期，避免连续触发。

GUI 参数（"视线跳转" 面板）：`触发距离(px)` 为文本输入框，输入合法正整数后立即生效；`冷却时间(ms)` 为滑块实时调节。

相关配置：

```python
gaze_jump_jump_threshold = 300
gaze_jump_cooldown_ms = 2000
gaze_jump_min_user_move = 5
```

### 4. 视线跟随（`gaze_follow`）

**触发逻辑**：当用户一段时间没有主动操作鼠标时，光标自动缓动跟随视线位置；检测到用户主动移动鼠标后暂停自动跟随，用户停止操作超过 `idle_seconds` 后恢复跟随。

相关配置：

```python
gaze_follow_idle_seconds = 3.0
gaze_follow_user_move_pixel = 3
gaze_follow_step_interval_ms = 30
gaze_follow_ease = 0.35
```

GUI 参数（"视线跟随" 面板）：`操作后暂停(秒)`、`跟随顺滑度`，均为滑块实时生效。

### 5. 视线滑翔（`gaze_glide`）

**触发逻辑**：当用户移动鼠标时，系统会判断：

1. 当前鼠标位置与视线位置是否足够远
2. 鼠标移动方向是否与 "当前鼠标位置 → 视线位置" 的方向一致
3. 两个方向的夹角余弦值是否超过阈值

如果满足条件，则对用户这一次鼠标移动附加额外相对位移，实现 "朝视线方向滑翔加速"。

速度系数近似为：

```text
factor = 1 + (max_multiplier - 1) × dist_factor × dir_factor
```

其中：

- `dist_factor`：距离越远越接近 1，越接近视线越接近 0
- `dir_factor`：方向越一致越接近 1，不一致时为 0
- `max_multiplier`：最大加速倍数

GUI 参数：`加速倍数`，滑块实时调节，当前范围为 `5.0x ~ 30.0x`。除了加速倍数，同一个“视线滑翔”面板里还可调节：减速起始距离、全速加速距离、减速陡峭度。

相关配置：

```python
gaze_glide_max_multiplier = 5.0       # 最大加速倍数（滑块可调）
gaze_glide_near_threshold = 300       # 距视线 <= 该值 → 不加速
gaze_glide_far_threshold = 500        # 距视线 >= 该值 → 距离因子取最大
gaze_glide_cos_threshold = 0.6        # cos <= 该值视为方向不一致
gaze_glide_stroke_reset_ms = 200      # 鼠标停顿超过此时长 → 重置 stroke 起点
gaze_glide_min_stroke_len = 20        # stroke 过短时暂不启用方向判定
gaze_glide_dist_exponent = 2.0        # 距离因子曲线指数，>1 时近处衰减更快（急刹车）
gaze_glide_overshoot_anchor_ratio = 0.5  # 越界锚：extra 位移最多落到“距目标 near*ratio”处
```

**实现注意点**（也是踩坑经验）：

- 使用 `pyautogui.move(dx, dy)` 做**相对位移叠加**，而非 `moveTo(x, y)` 绝对坐标跳转，避免事件队列滞后导致鼠标被拉回。
- 使用**实时光标位置** `pyautogui.position()` 计算距离和视线方向，减少快速手动移动时的竞态问题。
- 近距离不加速，方便在目标附近精细定位。

---

## 模型设计思路

上一章讲了"怎么用"，这一章讲"为什么这样设计"。这是理解 FollowMyGaze 的核心——也是它作为深度学习实验平台价值最高的部分。

视线预测本质是一个**多因素回归**问题：屏幕上的注视点，既取决于**头部相对屏幕的姿态**（头怎么摆的），又取决于**眼球相对头部的偏转**（眼珠往哪儿看）。这两个因素在物理上是**加性叠加**的：

```text
gaze_on_screen ≈ f_head(头部姿态) + f_iris(虹膜偏转)
```

FollowMyGaze 的模型（`gaze_feature_based_model.py::GazeMoE`）不把这两个因素揉在一起端到端硬回归，而是**显式地把两个因素拆成两个专家**，再用一个**基于物理先验的门控**决定 "这一帧要不要把虹膜项加进来"。这既符合视线的物理直觉，也大幅降低模型泛化到不同头姿 / 眼动状态时的漂移。

下面从「特征 → 网络 → 门控 → 损失 → 平滑 → 部署」六个层次展开。

### 1. 输入特征设计（约 124 维）

特征提取由 `gaze_feature_extractor_mac.py::extract_features_from_image` 完成，基于 MediaPipe FaceLandmarker 输出的 3D landmarks + 头部姿态矩阵。可分为以下几组：

**A. 人脸位置与大小（idx 0~4）**
- 人脸包围盒 `xmin, ymin, xmax, ymax`
- 人脸像素面积 `face_area`
- 作用：粗略反映人相对摄像头的距离与横向位置

**B. 头部姿态角（idx 5~7）**
- Pitch（俯仰）、Yaw（偏航）、Roll（翻滚）
- 从 MediaPipe 输出的旋转矩阵反解得到
- 作用：`f_head` 的核心输入，头姿一旦确定，"看屏幕正中的默认落点" 就大致定了

**C. 左眼 / 右眼特征（idx 8~23）**
每只眼独立提取：
- `rel_x, rel_y`：虹膜中心相对眼眶几何中心的偏移
- `iris_x, iris_y`：虹膜绝对坐标
- `openness`：眼睛开合度（防止闭眼时输入噪声太大）
- `ratio_x, ratio_y`：虹膜在眼眶横 / 纵向的比例位置（对眼睛大小归一）
- `iris_aspect`：可见虹膜的宽高比（半闭眼、侧脸时会变化，可作为置信度线索）

**D. 双眼辐辏（idx 24）**
- `eye_l_rel_x - eye_r_rel_x`：双眼虹膜水平偏移差，反映是否在看近处（辐辏信号）

**E. 3D 视线方向向量（idx 26~31）**
- 利用 MediaPipe 提供的 z 坐标，分别计算左 / 右眼的 3D gaze 单位向量
- 相比 2D 相对偏移，3D 方向对头部旋转更鲁棒

**F. 3D 瞳孔间距 IPD（idx 32）**
- 双眼虹膜的 3D 距离
- 强相关于人到摄像头的距离；也可以作为 "归一化尺度" 参考

**G. 关键 landmark 坐标**
- 眼睛轮廓、虹膜边缘、鼻尖、眉心等一组 landmark 的 `(x, y)`
- 提供未加工的原始几何信息，让网络自己学习二次特征

最终拼接成一个约 124 维的向量作为模型输入。

**设计动机**：不做重的图像 CNN，而是把可解释的、维度较低的几何+姿态特征喂给一个轻量 DNN。这样做的好处：

- 训练数据需求量小（个人采集通常几千个样本级别）
- 推理快，能实时跑
- 参数直觉清晰，异常时容易 debug
- 用户换发型 / 戴口罩 / 光照变化时依然稳健（MediaPipe 已经吸收了大量图像变化）

### 2. GazeMoE：门控混合专家（Mixture of Experts）

有了 124 维特征，接下来的问题是：**怎么把这些特征喂给网络，才能保留物理直觉？**

答案是 MoE：

```text
       ┌───────────────────────────────────────────┐
       │                x0 (B, 124)                │
       └──────────┬──────────────────────┬─────────┘
                  │                      │
             ┌────▼─────┐          ┌─────▼────┐
             │ HEAD 专家 │          │ IRIS 专家 │
             │ (同构)    │          │ (同构)    │
             └────┬─────┘          └─────┬────┘
                  │                      │
                  ▼                      ▼
              y_head (B,2)          y_iris (B,2)
                  │                      │
                  │        ┌─────────────┘
                  │        │
                  │        ▼   gate ∈ {0,1}（STE）
                  │   gate·y_iris
                  │        │
                  └───────►+
                           │
                           ▼
                       y_final = y_head + gate * y_iris
```

**两个同构专家 + 一个门控**：

- **Head 专家**：负责把 "头姿+人脸位置" 映射到 "如果虹膜完全居中，注视点会落在哪"
- **Iris 专家**：负责在 Head 专家给出的基准位置上，加上 "因为眼球偏转带来的位移量"
- **Gate**：决定这一帧要不要把 Iris 贡献量加上去

关键选择：**两个专家结构完全同构**，都能看到全部 124 维输入。不做人工特征筛选（比如 "只把头姿给 head 专家"），而是让门控和多任务损失去教它们各自的分工。

### 3. 专家网络结构

每个专家的内部结构：

```text
x0 (124) ──► Linear(124→128) + BN + ReLU + Dropout ─┐
             │                                       │
             └► CrossNet(124, 1 层) ─────────────────┤
                                                     ▼
                                          concat → (B, 252)
                                                     │
                              ResBlock(252 → 128) ──►│
                              ResBlock(128 → 64)  ──►│
                              ResBlock(64  → 32)  ──►│
                              ResBlock(32  → 16)  ──►│
                                                     ▼
                                                  (B, 16)
                                                     │
                                          Linear → (B, 2)
```

组件说明：

**a. 输入变换** —— `Linear + BN + ReLU + Dropout`：常规特征映射到 128 维隐空间。

**b. CrossNet（Deep & Cross Network 的 Cross 层）**
- 每层做 `x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l`
- 显式建模特征间的**低阶交叉**（例如 pitch × iris_rel_x 这种 "头往下看时虹膜偏移的意义" 这种交互）
- 只用 1 层，避免过参数化

**c. 4 层 ResidualBlock**
- 每个 block：`Linear + BN + ReLU + Dropout + Linear + BN` + 残差 shortcut
- 通道逐步收缩 `128 → 64 → 32 → 16`
- 残差和 BN 都是为了在小数据集上稳定训练

**d. Iris 专家的 anchor 拼接**
- Iris 专家主干输出 (B, 16) 后，会拼上 `y_head.detach()` (B, 2)
- 再过输出层得到 `y_iris`
- 语义：让 Iris 专家知道 "head 专家现在把基准点定在哪儿"，只需要输出**相对偏移量**
- `.detach()` 保证反传不会通过这里污染 y_head 的语义

### 4. Gate 路由：基于物理先验的二值门控

网络结构决定了两个专家能"分工"，Gate 决定了两个专家在**每一帧**如何分工。

Gate 的核心思想：

> 当虹膜相对眼眶几乎没偏移时（人在直视头姿默认方向），y_iris 应该被强制归零，让 y_head 独立承担预测；
> 当虹膜有明显偏移时，y_iris 才被 "开启"，与 y_head 相加。

这是一个**近似 0/1 的开关**，不是 [0, 1] 的软权重。为此 GazeMoE 用了一套自适应二值化方案：

**a. Prior offset：`||rel||`**
- 选取左 / 右眼的 `(rel_x, rel_y)` 共 4 维作为门控输入
- 先经过一个 `BatchNorm(affine=False)` 去 DC 分量（消除个体眼型 bias）
- 计算 L2 范数 `||rel||`，物理含义是 "两只眼虹膜相对眼眶中心的总偏移量"

**b. Otsu 自适应阈值**
- 每个训练 batch，用 Otsu 算法在 `||rel||` 上找**最大类间方差**的分界值
- 用 EMA(0.99) 平滑到 `threshold_ema`，作为全局门控阈值
- 好处：不用手调 "多小算居中" 这个超参，模型自己学

**c. Sharpness = K / std(||rel||)**
- 用 `||rel||` 的 EMA 标准差调整 sigmoid 陡度
- 数据分布散 → sigmoid 平缓；数据分布集中 → sigmoid 陡峭
- 保证 sigmoid 输出接近 0 / 1 的极端值

**d. Straight-Through Estimator（STE）二值化**

```python
soft_gate = sigmoid((||rel|| - threshold) * sharpness)
hard_gate = (soft_gate > 0.5).float()
prior_gate = hard_gate + (soft_gate - soft_gate.detach())
```

- 前向：`hard_gate` 是严格的 0 或 1
- 反向：梯度走 `soft_gate` 的路径，可以正常训练
- 效果：推理时门是"干净的开关"，训练时又能学

**e. Residual gate**
- 一个可学的小 MLP 输出 `[-1, 1]`（Tanh）
- 用来做**微调**，防止 prior 一刀切错
- 当前 `GATE_RESIDUAL_SCALE = 0.0`，等价于纯 prior 门控（简化版）

### 5. 多任务损失：让专家各司其职

有了 gate 之后还有一个隐患：**如果损失只有 `MSE(y_final, target)`**，网络可能把大部分预测放在 y_head 或 y_iris 任一侧，另一个变废；或者 y_iris 学到一些 "不该属于虹膜偏转的东西"。

所以 `GazeController._compute_loss` 用了三项损失，把两个专家的角色**明确写进 loss**：

```text
loss_total = 1.0 * loss_final
           + 0.8 * loss_head_aux
           + 0.5 * loss_iris_aux
```

**loss_final（主目标）**
```python
loss_final = MSE(y_head + gate * y_iris, target)
```

**loss_head_aux（虹膜居中时监督 y_head）**
```python
loss_head_aux = mean((1 - gate.detach()) * (y_head - target)^2)
```
- 只在 gate ≈ 0（虹膜居中）时激活
- 语义："虹膜没动，标签的注视点就应该等于头姿贡献。此时逼 y_head 直接学 label。"
- 这批数据是**最干净的 y_head 监督信号**

**loss_iris_aux（虹膜偏转时学贡献，虹膜居中时约束归零）**
```python
loss_iris_aux = mean( gate.detach()      * (y_head.detach() + y_iris - target)^2
                    + LAMBDA * (1 - gate.detach()) * y_iris^2 )
```
- 第一项：gate=1 时（虹膜偏转），要求 y_iris 补足 y_head 到 label 的差
- 第二项：gate=0 时（虹膜居中），强制 y_iris ≈ 0（不要瞎输出）
- `LAMBDA = 1.0` 控制归零项的力度

**为什么用 `.detach()`？**

- `gate.detach()`：loss 的权重不通过 gate 反传，避免为了降 loss 让 gate 乱跳
- `y_head.detach()` 在 iris_aux：y_iris 只学 "补差量"，不通过 iris_aux 污染 y_head

### 6. 推理阶段的 EMA 平滑

`GazeController.predict_gaze` 里对每帧输出再做一次 EMA：

```python
ema_x = alpha * raw_x + (1 - alpha) * ema_x   # alpha = 0.9
```

**动机**：模型逐帧预测不可避免有几像素到十几像素级别的抖动，直接驱动红点 / 光标会不停颤动。EMA 用一个非常轻的滤波（alpha=0.9，等价于 ~10 帧滑动均值的滞后）把抖动压下去。

### 训练/推理超参一览

| 项目 | 值 | 说明 |
|---|---|---|
| Optimizer | Adam | `lr=0.001` |
| Loss | 多任务加权 MSE | 权重 1.0 / 0.8 / 0.5 |
| Online batch size | 2048 | 自动训练 |
| Online epoch | 40 | 自动训练 |
| Offline batch size | 2048 | 手动训练 |
| Offline epoch | 400 | 手动训练 |
| Auto-train threshold | 1024 samples | 累计 1024 新样本触发一次 |
| EMA alpha (inference) | 0.9 | 越小越平滑越滞后 |
| Otsu bin 数 | 50 | Gate 阈值统计 |
| Threshold EMA momentum | 0.99 | Gate 阈值平滑 |
| K_SHARP | 5.0 | sigmoid 陡度系数 |
| GATE_RESIDUAL_SCALE | 0.0 | 当前纯 prior gate |

---

## 训练与推理机制

模型设计再好，也要能在**实时系统**里跑起来。这一章讲工程细节。

### Shadow Model 训练：避免线程竞态

推理线程（GUI 摄像头刷新）和训练线程（手动/自动训练）会同时访问同一个 `nn.Module` 实例，会遇到：

- BatchNorm 的 `running_mean/running_var` 在训练时被更新，推理线程读到不一致状态
- 训练过程中权重是 "半成品"，推理线程读到会给出跳变的预测
- `forward` 在训练模式和评估模式下返回不同数量的 tensor，另一线程若正好切换模式会 unpacking 出错

`train_and_predict_dnn.py::GazeController.train_model` 的做法：

1. `shadow = GazeMoE().to(device)`：全新一份模型
2. `shadow.load_state_dict(self.model.state_dict())`：warm start，从当前推理模型继承权重
3. 训练全部跑在 `shadow` 上，`self.model` 完全没被动过
4. 训练结束后 `shadow.eval()`，`torch.save(shadow.state_dict(), ...)` 落盘
5. `self.model = shadow`：Python 属性赋值是原子操作，下一帧推理无缝切换

**副作用**：训练期间会双倍占用一份模型显存 / 内存；对当前规模的小模型完全可以接受。

### 推理数据流

```text
摄像头 (30fps)
   │
   ▼
GazeFeatureExtractor.extract_features_from_image  ── 后台线程池
   │  → 124 维特征
   ▼
GazeController.predict_gaze                       ── 后台线程池
   │  → 预测视线 (x, y) + EMA 平滑
   ▼
GlobalInfo.red_dot_x / red_dot_y
   │
   ├──► RedDotOverlay._tick (30ms 刷新)
   └──► CursorModeManager (gaze_jump / follow / glide)
```

### 采样与自动训练

```text
用户左键点击
   │
   ▼
utils.start_listening_click_dnn.on_click
   │  → save_sample(features, coords)
   ▼
GazeDataset.save_sample
   │  → features_new / labels_new / df 累加
   │  → 更新 UI 样本计数
   │
   ├── 每 auto_train_threshold 样本
   │      │
   │      ├──► _auto_save_to_local（追加到 samples.pkl）
   │      └──► _trigger_auto_train（后台线程 shadow-training）
   │
   └── 程序退出
          └──► save_current_sample_to_local
```

---

## 配置说明

主要配置在 `global_info.py` 中。

### 训练配置

```python
online_training_batchSize = 2048
online_training_epoch = 40

offline_training_batchSize = 2048
offline_training_epoch = 400
```

- 在线自动训练使用 `online_training_epoch`
- 手动全量训练使用 `offline_training_epoch`

### 自动训练配置

```python
auto_train_threshold = 1024
enable_auto_train = True
```

每累计 1024 个新样本触发一次自动保存和后台训练。

### 样本上限

```python
sample_upper_limit = 100000
```

本地样本超过上限时会保留最新样本，丢弃最旧样本。

### 模型与数据目录

```python
path_dir = os.path.join(os.path.expanduser("~"), "FollowMyGaze")
```

数据与模型保存在**用户主目录**下（打包成 `.app` 后工作目录是 `/`，必须用绝对路径，不能依赖相对路径）：

```text
~/FollowMyGaze/gaze_model_resnet.pth   # 训练好的模型权重
~/FollowMyGaze/samples/samples.pkl     # 采集的训练样本
~/FollowMyGaze/user_config.json        # GUI 参数持久化
```

---

## 项目结构

```text
FollowMyGaze/
├── main.py                         # 程序入口：初始化摄像头、模型、数据集、GUI
├── gui.py                          # Tkinter GUI、摄像头刷新、训练按钮、模式选择、窗口图标
├── global_info.py                  # 全局运行状态与可配置参数
├── user_config.py                  # GUI 参数持久化（保存/加载配置文件）
├── data_process_dnn.py             # DNN 特征样本数据集、采样保存、自动训练触发
├── train_and_predict_dnn.py        # GazeController：训练、推理、模型保存/加载
├── gaze_feature_based_model.py     # GazeMoE 视线预测模型结构
├── gaze_feature_extractor.py       # 通用 MediaPipe 特征提取器
├── gaze_feature_extractor_mac.py   # macOS 版本 MediaPipe 特征提取器
├── gaze_cursor_modes.py            # 视线跳转/跟随/滑翔等鼠标交互 controller
├── utils.py                        # 图像缩放、点击监听、Alt 光标随动、资源路径等工具
├── convert_icon.py                 # 把图片转换为 PNG/ICNS/ICO 图标
├── face_landmarker.task            # MediaPipe FaceLandmarker 模型资源
├── main.spec                       # PyInstaller 打包配置（含 icon.icns）
├── cnn_model.py                    # 早期 CNN 模型实验代码
├── data_process.py                 # 早期图像数据处理代码
├── train_and_predict.py            # 早期 CNN 训练/推理代码
├── test.py / tmp.py                # 临时测试脚本
├── build/                          # 打包构建产物
└── dist/                           # 打包输出产物
```

---

## 常见问题

**1. 摄像头画面不显示**

检查：
- 摄像头是否被其他程序占用
- macOS 是否给终端 / IDE / Python 授权摄像头权限
- `cv2.VideoCapture(0)` 是否对应正确摄像头

**2. 鼠标/键盘监听无效，或"前台能采集、后台不采集"**

macOS 的 `CGEventTap` 需要**两个独立权限**，请**全部**授予运行程序的终端 / IDE / Python（或打包后的 `.app`）：

```text
系统设置 -> 隐私与安全性 -> 辅助功能    （控制鼠标：pyautogui）
系统设置 -> 隐私与安全性 -> 输入监控    （监听点击：pynput）
```

- **红点/视线跟随能动，但采集不到点击样本** → 缺「输入监控」。
- **应用在前台能采集，切到后台就停** → 典型的缺「输入监控」表现：未授权时全局监听被降级为只接收发给本应用的前台事件。
- **打包成 `.app` 后授权失效** → `.app` 是独立身份，需单独授权；且 ad-hoc 签名每次重新打包指纹变化，会导致授权失效，需重新逐一授权。
- 授权后请**完全退出并重启**应用：`CGEventTap` 只在启动时创建一次，授权必须先于启动才生效。

否则 `pynput` / `pyautogui` 可能无法监听或控制鼠标键盘。

**3. 红点追踪偏差很大**

通常是样本不足或样本分布不均导致。建议：
- 在屏幕四角、边缘、中间区域都采样
- 每个区域多次点击
- 保持头部和坐姿相对稳定
- 训练后再打开红点观察

**4. 视线滑翔出现抽动或过快**

可以调低 `gaze_glide_max_multiplier` / `gaze_glide_far_threshold`，或调高 `gaze_glide_cos_threshold` / `gaze_glide_min_stroke_len` / `gaze_glide_near_threshold`。GUI 里也可以直接调整 "视线滑翔加速倍数"。

**5. PyAutoGUI FailSafe 报错**

项目已在 `gaze_cursor_modes.py` 中关闭：

```python
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
```

并对屏幕边缘做了坐标 clamp，减少触发角落 FailSafe 的概率。

---

## 打包

### 生成应用图标（可选）

准备一张图片（如 `aaa.jpg`），运行脚本一键生成三种格式的图标：

```bash
python convert_icon.py
```

产出：

- `icon.png`：Tkinter 运行时窗口图标（`gui.py` 启动时自动加载）
- `icon.icns`：macOS 应用图标（`.icns` 只能在 macOS 上通过 `iconutil` 生成）
- `icon.ico`：Windows 可执行文件图标

如需更换源图片，修改 `convert_icon.py` 顶部的 `src = "aaa.jpg"`。

### PyInstaller 打包

项目包含 `main.spec`，可使用 PyInstaller 打包：

```bash
pyinstaller main.spec
```

注意事项：

- 打包时需要确保 `face_landmarker.task` 与 `icon.png` 被包含到最终产物中（已在 `main.spec` 的 `datas` 中声明）。
- macOS 应用图标由 `main.spec` 中 `BUNDLE(..., icon='icon.icns')` 指定；Windows 可在 `EXE(..., icon='icon.ico')` 指定。
- `utils.resource_path()` 已兼容 PyInstaller 的 `_MEIPASS` 路径。

---

## 开发建议

- 新增交互模式时，优先放在 `gaze_cursor_modes.py`，并通过 `CursorModeManager` 统一启停。
- 涉及 UI 更新时，后台线程不要直接操作 Tk 控件，应使用 `root.after(...)` 回到主线程。
- 涉及模型训练时，避免直接修改正在推理的模型实例，继续沿用 shadow model 模式。
- 涉及鼠标移动时，尽量区分：
  - `moveTo(x, y)`：绝对坐标，适合跳转
  - `move(dx, dy)`：相对位移，适合叠加加速
- 对 `pynput` 事件坐标要保持谨慎，快速移动时事件坐标可能落后于真实光标位置。

---

## 未来展望

FollowMyGaze 目前是一个能用、可玩、可实验的 MVP，但离"日常大规模稳定使用"和"通用视线研究平台"都还有距离。以下是几个方向，也是欢迎共建的地方。

### 模型层面

- **头姿分支 / 虹膜分支的解耦评估**：目前只看最终 `y_final` 的误差；可以加评估脚本，分别看 `y_head` 单独、`y_iris` 单独、以及不同 gate 状态下的误差分布，用来诊断专家是否真的学到了各自的角色。
- **门控残差实验**：把 `GATE_RESIDUAL_SCALE` 从 0.0 打开，看能否用小 MLP 修正 Otsu 一刀切的边界样本。
- **多任务权重的自适应**：现在 `1.0 / 0.8 / 0.5` 是手工调的，可以引入 GradNorm / uncertainty weighting 自动调。
- **超越 MediaPipe 特征**：把摄像头图像作为额外分支，比如小型 CNN 抽 patch 特征 concat 到几何特征上，形成 "几何 + 图像" 双流模型。
- **时序建模**：现在每帧独立预测再做 EMA。可以尝试直接用 GRU / TCN 建模一小段时序窗口。

### 交互层面

- **视线 + 语音 / 手势混合触发**：视线定位 + 手势/语音确认，进一步降低手部使用。
- **应用感知的智能模式**：在代码编辑器里保守跟随，在浏览器里滑翔更激进。
- **可视化调试面板**：实时显示 gate 值、专家输出、EMA 前后偏差，边用边看。

### 工程层面

- **跨平台**：`gaze_feature_extractor_mac.py` 单独抽出，做 Windows / Linux 版本。鼠标控制部分也一样。
- **打包与自启**：让整个应用可以像输入法一样一键安装、开机自启、后台常驻。
- **用户 profile 管理**：多个用户 / 多种坐姿分别存 checkpoint，切换时自动加载对应模型。
- **联邦 / 匿名数据聚合**：在保护隐私的前提下，用多人数据训练一个更好的 base model，个人再 fine-tune。

### 研究层面

- **注意力可视化**：把视线数据用于阅读研究、UI 可用性测试、心流指标。
- **fatigue 检测**：从眼动模式变化推断用户疲劳程度。
- **可访问性辅助**：为运动能力受限的用户提供纯视线的鼠标替代。

如果你对以上任一方向感兴趣，欢迎提 issue、开 PR，或直接拿这个项目做你自己的实验分支——反正它一开始就是为"改一版立即验证"设计的。

---

## License

当前仓库未声明 License。如需开源发布，建议补充明确的许可证文件。
