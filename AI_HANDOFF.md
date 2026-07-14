# DynamicVLA 真机部署二次开发完整交接文档

> 最后更新：2026-06-24  
> 仓库：`/home/yuling/work_space/repos/DynamicVLA`  
> 真机部署目录：`deploy/`  
> 当前状态：双相机、Piper CAN、模型加载、异步推理、LAAS调度、shadow与受控execute均已实机贯通。默认执行后端为`host_ik_move_j`：主机侧受限多初值IK已实现并通过26项离线测试，但尚未完成该后端的真机运动验收。系统仍无完整碰撞规划、轨迹时间参数化和严格跟踪误差联锁，execute只能在现场监护下用于受控验收，不是生产安全控制器。

## 0. 阅读顺序与当前权威状态

### 0.1 内容优先级

本文档同时保留开发历史和当前设计。为避免把旧阶段结论当成现状，统一按以下顺序判定：

1. 当前源码、YAML和测试结果。
2. 本节的当前状态摘要。
3. 第27节“三种IK/执行方案总览与切换设计”。
4. 第26节的主机IK实现细节。
5. 第1—24节为基础设计和早期阶段记录；第25节为MOVE P实机阶段的历史记录。它们不得覆盖后续权威章节。

若文档与当前源码不一致，必须先检查`git diff`、实际配置和测试，不得仅凭文档启用真机运动。

### 0.2 当前实现与验证状态

| 项目 | 当前状态 |
|---|---|
| 默认后端 | `host_ik_move_j` |
| Piper固件IK | `firmware_move_p`代码保留，仅作诊断和受控回退 |
| MoveIt/KDL | `moveit_kdl`尚未接入`deploy/`，不得表述为已实现 |
| 主机IK | SciPy受限多初值数值IK，Piper官方FK，显式误差、限位余量和关节跳变检查 |
| 控制下发 | MOVE J + `JointCtrl`；夹爪使用独立`GripperCtrl` |
| 状态联锁 | execute要求CAN + MOVE J + normal；故障后停止策略动作并保持当前关节 |
| LAAS | 首个accepted chunk从`action[0]` bootstrap；后续chunk才执行常规过期跳步 |
| 测试 | 文档最新记录为`26 passed`；修改后必须以实际测试输出为准 |
| 真机验收 | 旧MOVE P链路已真机运动；新`host_ik_move_j`后端尚待真机运动验收 |
| 安全定位 | shadow为默认；execute仅用于受控验收，必须现场监护 |

### 0.3 尚未实现或未完成验收

- 桌面、自碰撞、相机、夹爪和线缆包络的完整碰撞检查。
- OMPL或等价路径搜索、轨迹时间参数化与连续速度/加速度约束。
- `moveit_kdl`后端、PlanningScene和异步latest-only MoveIt客户端。
- `host_ik_move_j`的shadow日志验收、单步真机验收和连续真机验收。
- 严格的命令—反馈跟踪误差、卡滞、速度和加速度联锁。
- 真机示范采集器以及针对当前新起点和双相机域的post-training。

---

## 1. 文档目的

本文面向接手本项目的AI或开发者，目标是提供无需重新考古即可继续开发的完整上下文，包括：

1. DynamicVLA论文和原版仓库解决的问题。
2. 当前真实硬件、SDK、checkpoint和相机角色。
3. 原版仿真坐标、动作语义以及真机Piper坐标之间的关系。
4. `deploy/`全部模块的职责、接口、线程和数据流。
5. Continuous Inference和LAAS在当前实现中的具体行为。
6. 当前安全机制的边界、未实现部分和禁止绕过的检查。
7. 已完成的实机验证、日志结果和当前模型输出异常的根因。
8. 下一阶段二次开发的推荐顺序和必须保持的工程不变量。

本文是交接总入口；用户操作手册见 `deploy/README.md`，实验排障和记录模板见 `deploy/DEBUGGING.md`。

---

## 2. 项目背景

### 2.1 上游项目

- 项目名称：DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation
- 上游仓库：https://github.com/hzxie/DynamicVLA
- 本地论文：`2601.22153v1.pdf`
- 论文模型规模：约0.4B参数，主体为SmolLM2-360M、FastViT视觉编码器和扩散/流式动作专家。
- 动作chunk长度：20。
- 默认时间观测窗口：`{t-2, t}`，当前配置为`history_indices: [-2, 0]`。
- DOM数据：论文报告20万仿真episode、2000真机episode、2800场景、206种物体。

论文的三个核心设计：

1. 紧凑VLA架构：降低多模态推理延迟。
2. Continuous Inference：推理和当前动作chunk执行并行，避免执行完一整个chunk后才开始下一轮推理。
3. LAAS（Latent-aware Action Streaming）：新chunk完成时跳过已经因推理延迟而过期的动作，只接管当前及未来动作。

### 2.2 “88 Hz”的正确解释

论文附录同时给出：

- 16层模型单个chunk推理时间约`0.226 s`（RTX A6000）。
- 每个chunk包含20个动作点。
- `20 / 0.226 ≈ 88.5`个动作点/秒。

因此论文中的约88 Hz应理解为chunk内动作点吞吐率，不是每秒完成88次完整VLA重规划。完整chunk推理频率约为`1 / 0.226 ≈ 4.4 Hz`。

当前机器实测最新shadow：

- 平均推理时间约`0.2825 s/chunk`。
- p95约`0.3161 s`。
- 完整重规划频率约3.5 Hz。
- 理论动作点吞吐约`20 / 0.2825 ≈ 70.8 Hz`。
- 真机控制循环配置为25 Hz，因此实际调度最多每秒消费25个动作点。

### 2.3 原版仓库不包含的部分

上游公开代码主要面向Isaac Lab仿真、训练和评测，没有提供可直接用于当前硬件组合的完整真机部署程序。当前`deploy/`为本项目新增，实现：

- Orbbec Gemini 305 RGB采集。
- Intel RealSense D435i RGB采集。
- AgileX Piper CAN反馈和控制适配。
- 模型异步加载与推理。
- 时间戳感知的LAAS动作调度。
- Piper SDK坐标到训练模型TCP坐标的双向SE(3)转换。
- shadow/execute隔离、笛卡尔安全过滤和JSONL日志。
- 只读硬件诊断、坐标验证和日志汇总工具。

论文真机数据采集中的“real-world simulator”还包含EfficientTAM、多视角分割、三角化物体3D位置、速度估计和自动状态机控制。**这些对象6D状态估计功能当前没有实现**；当前部署只向VLA提供双RGB、末端状态和语言任务。

---

## 3. 当前硬件与软件

### 3.1 硬件角色

| 硬件 | 当前角色 | 模型特征名 |
|---|---|---|
| Orbbec Gemini 305 | 腕部RGB相机 | `observation.images.wrist_cam` |
| Intel RealSense D435i | 固定第三视角RGB相机 | `observation.images.opst_cam` |
| AgileX Piper | 6轴机械臂和电动夹爪 | 6维末端state、7维action |
| USB-CAN适配器 | Piper CAN 1 Mbps | `can0` |

注意：用户最初设想曾与当前映射相反，最终确认的配置是Gemini 305为腕部、D435i为固定第三视角。YAML和代码已按最终映射配置。

### 3.2 Python环境与依赖

- Conda环境：`dynamicvla`
- Python：3.10
- PyTorch实测：2.8.0+cu128
- 真机附加依赖：`deploy/requirements.txt`
  - `pyorbbecsdk2==2.1.1`
  - `pyrealsense2==2.58.2.10647`
  - `piper-sdk==0.6.1`
- 坐标变换依赖SciPy，主仓库`requirements.txt`已包含。

### 3.3 udev与CAN

- 相机udev规则用于允许非root用户访问USB设备；验证SDK导入不需要插相机，但验证枚举、数据流和权限必须连接设备。
- Piper使用SocketCAN，已验证：
  - 接口：`can0`
  - bitrate：`1000000`
  - 状态：`UP, LOWER_UP, ERROR-ACTIVE`

典型CAN初始化：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

---

## 4. Checkpoint契约

### 4.1 当前checkpoint

```text
/home/yuling/work_space/repos/DynamicVLA/dynamic-vla-DOM
```

目录包含：

- `config.json`
- `model.safetensors`

关键配置：

```text
n_obs_steps       = 2
history           = [-2, 0]
chunk_size        = 20
n_action_steps    = 20
use_delta_action  = true
rotation          = euler
image input       = 3×360×480
```

### 4.2 输入

模型实际需要：

```text
observation.state: [x, y, z, rx, ry, rz]       shape [6]
observation.images.wrist_cam                   shape [3,360,480]
observation.images.opst_cam                    shape [3,360,480]
task: string
```

state表示`base_link -> model_tcp`的绝对位姿：

- XYZ单位：米。
- Euler单位：弧度。
- Euler约定：SciPy小写`"xyz"`，即静态轴/外旋XYZ。
- 数据预处理将Euler X、Z包装到`[0, 2π)`，Y保留SciPy主值范围。

模型**不输入**：

- 六关节角。
- 夹爪反馈。
- 深度图。
- IMU。
- 相机外参。
- 对象6D位姿或速度。

`RobotState.model_vector()`会生成7维`[pose6, gripper]`，但`DynamicVLAWorker.fit_feature_dimension()`根据checkpoint声明裁剪为6维，因此当前夹爪反馈不会进入模型。

### 4.3 输出

输出shape为`[20, 7]`：

```text
[x, y, z, rx, ry, rz, gripper]
```

checkpoint设置`use_delta_action=true`。模型直接生成前六维增量，`DynamicVLAWorker`按照原版逻辑逐分量加到最新观测state：

```text
absolute_action[..., :6] = predicted_delta[..., :6] + latest_state[:6]
```

安全层和Piper适配层看到的是转换后的绝对目标。Euler也是逐分量相加，这不是严格SE(3)旋转复合，但必须和原版训练/推理语义保持一致。

夹爪训练标签：

- `+1 = open`
- `-1 = close`

神经网络可能输出任意连续值，当前安全层以0为阈值二值化。

### 4.4 checkpoint归一化统计

从`model.safetensors`读取到：

```text
state mean =
[0.384749, -0.001231, 0.172404, 3.136002, -0.117838, 2.910169]

state std =
[0.125090, 0.152252, 0.106291, 0.059157, 0.239787, 2.818949]

action mean =
[0.385432, -0.001203, 0.162089, 3.138075, -0.074502, 2.891039, 0.308464]

action std =
[0.128269, 0.155269, 0.105443, 0.058275, 0.200727, 2.819702, 0.951236]
```

这些统计对判断真机state是否分布外非常重要。不能仅以“位于机械臂物理工作空间”为依据判断模型可用。

2026-06-29 对 `/data/datasets/DOM/data/chunk-000` 已下载的 921 个 parquet、183266 行做实测，`action[:6] - observation.state[:6]` 得到：

```text
delta action mean =
[0.001576, 0.000079, -0.007462, 0.001280, 0.027347, -0.014355, 0.226501]

delta action std =
[0.035303, 0.053547, 0.045835, 0.030943, 0.080895, 1.894453, 0.974011]
```

当前真机 deploy runtime 在 `DynamicVLAWorker` 中覆盖 `normalize_targets/unnormalize_outputs.buffer_action` 为上述 delta-action 统计；官方 checkpoint 文件本身未改。该覆盖只适用于部署实验，完整 DOM 或后训练数据重算后应重新替换。

---

## 5. 坐标系完整定义

### 5.1 原版Isaac坐标

原版仿真首先将世界坐标转换到机器人根坐标：

```text
p_base = R_world_base^T × (p_world - p_world_base)
q_base_ee = inverse(q_world_base) ⊗ q_world_ee
```

所以模型使用机器人`base_link`坐标，不使用Isaac world坐标。轴方向：

- +X：机械臂正前方。
- +Y：机械臂左侧。
- +Z：向上。

### 5.2 原版仿真TCP

原版Piper URDF存在：

```text
link6 -> gripper_base = Rz(π)
```

Isaac FrameTransformer追踪的`end_effector`位于：

```text
gripper_base局部 +Z 0.1334 m
```

Isaac DLS IK控制配置使用`gripper_base`并设置约`+Z 0.137 m`的控制偏移。数据state/action的追踪点以0.1334 m定义为准。

### 5.3 Piper SDK坐标实测

Piper官方接口：

- `GetArmEndPoseMsgs()`返回XYZ（0.001 mm）和RX/RY/RZ（0.001°）。
- `GetFK("feedback")[-1]`返回joint6相对`base_link`的位姿。
- 使用FK前必须调用`EnableFkCal()`。

实机静止采集100个样本：

```text
EndPose median mm/deg:
[47.688, -2.917, 166.846, 173.111, 66.826, 168.489]

FK joint6 median mm/deg:
[48.634, -2.996, 168.582, 173.353, 66.008, 168.712]

位置误差: 1.979 mm
旋转误差: 0.824°
状态: 100/100均为0x00 normal
```

由此确认当前固件的EndPose可以按`base_link -> link6`处理。

### 5.4 当前双向变换

定义固定变换：

```text
T_sdk_model = T_link6_model_tcp
translation = [0, 0, 0.1334] m
rotation    = Rz(π)
```

反馈：

```text
T_base_model = T_base_sdk × T_sdk_model
```

命令：

```text
T_base_sdk = T_base_model × inverse(T_sdk_model)
```

实现在`deploy/devices/piper_frames.py`，只允许在设备边界转换一次：

- 策略、日志中的state、安全workspace全部使用model TCP。
- `GetArmEndPoseMsgs`和`EndPoseCtrl`边界使用SDK link6。
- 不允许在策略层或安全层再次补0.1334 m或Rz(π)。

当前实机EndPose转换后的模型TCP：

```text
position_m    = [0.170183, -0.011534, 0.114729]
euler_xyz_rad = [3.261828, -1.166334, 6.082280]
```

局部TCP偏移必须经过末端旋转，不能简单给base Z加0.1334。

### 5.5 相机坐标

相机仅提供RGB，当前模型没有数值相机外参输入。相机安装姿态仍然会通过图像域强烈影响模型：

- Gemini 305必须对应训练中的腕部视角。
- D435i必须对应训练中的固定第三视角。
- 名称对调不会报shape错误，但会造成严重语义错误。

原版仿真腕部相机挂载在`gripper_base`，局部位置约`[0.065, 0, 0]`，使用OpenGL相机约定。真机安装需要尽量复现视野、朝向和遮挡关系。

---

## 6. 总体运行架构

```text
Orbbec线程 ──RGB──> FrameBuffer(wrist_cam) ─┐
                                             │
RealSense线程──RGB──> FrameBuffer(opst_cam) ─┤
                                             ▼
Piper反馈线程──RobotState────────────> ObservationBuilder
                                             │  历史[-2,0]
                                             ▼
                                      PolicyObservation
                                             │ 单槽覆盖队列
                                             ▼
                                   DynamicVLAWorker线程
                                             │ ActionChunk[20,7]
                                             ▼
                                      ActionScheduler
                                      LAAS跳过过期前缀
                                             │ ScheduledAction
                                             ▼
                                        SafetyFilter
                                  workspace/步长/角度/夹爪
                                             │
                    shadow: 只记日志 <───────┴───────> execute
                                                     │
                                      model TCP逆变换到link6
                                                     │
                                  MotionCtrl_2 + EndPoseCtrl
                                                     │
                                      可选GripperCtrl
```

并发单元：

1. Orbbec采集线程。
2. RealSense采集线程。
3. Piper反馈线程。
4. DynamicVLA推理线程。
5. 主线程25 Hz控制循环。

所有设备/推理线程异常通过`error`字段传给主循环watchdog；主循环发现异常后退出并在`finally`停止设备。

---

## 7. `deploy/`目录与模块职责

```text
deploy/
├── AI_HANDOFF.md                 # 本文，AI二次开发总交接
├── README.md                     # 用户运行和架构说明
├── DEBUGGING.md                  # 排障、验收和实验记录
├── config.py                     # dataclass配置加载与校验
├── configs/
│   └── piper_gemini_d435i.yaml   # 当前硬件配置
├── common/
│   ├── messages.py               # 跨模块不可变消息结构
│   ├── latest.py                 # LatestValue与FrameBuffer
│   └── event_log.py              # 线程安全JSONL事件日志
├── devices/
│   ├── camera_base.py            # 相机线程抽象
│   ├── factory.py                # 相机驱动工厂
│   ├── orbbec_camera.py          # Gemini 305 RGB适配
│   ├── realsense_camera.py       # D435i RGB适配
│   ├── piper_frames.py           # SDK link6/model TCP双向变换
│   └── piper_robot.py            # Piper反馈、使能和命令适配
├── policy/
│   ├── observation_builder.py    # 双相机软件同步与时间历史堆叠
│   ├── inference_worker.py       # 模型异步加载/推理/delta转绝对
│   └── action_scheduler.py       # LAAS动作接管和过期处理
├── control/
│   └── safety_filter.py          # 当前笛卡尔安全过滤
├── tools/
│   ├── summarize_log.py          # events.jsonl统计
│   ├── verify_piper_frames.py    # EndPose/FK/model TCP只读验证
│   ├── move_to_training_start.py # 低速MOVE J回仿真初始关节，默认dry-run
│   ├── jog_piper_joint.py        # 单关节低速微动并保持最终位置
│   └── move_to_joint_zero.py     # 低速移动到现有六轴标定零位
├── tests/
│   ├── test_action_scheduler.py
│   ├── test_inference_worker.py
│   ├── test_piper_frames.py
│   └── test_safety_filter.py
├── diagnose.py                   # 不加载模型、不使能的联合设备诊断
├── run.py                        # runtime入口与生命周期
└── requirements.txt              # 三个真机SDK依赖
```

### 7.1 `common/messages.py`

- `CameraFrame`：相机名、序列号、设备帧号、设备时间戳、主机单调时间戳、RGB、可选depth。
- `RobotState`：
  - `joint_radians[6]`
  - `position_m[3]`，已经是model TCP
  - `euler_xyz_rad[3]`，已经是模型Euler约定
  - `gripper_m`
  - `feedback_hz`
- `PolicyObservation`：episode/index/timestamp、两路历史图像、历史state、task。
- `ActionChunk`：来源观测、完成时间、`[20,7]`动作及推理耗时。
- `ScheduledAction`：目标控制index、来源观测index和单个7维动作。

### 7.2 `common/latest.py`

- `LatestValue[T]`：线程安全单槽，发布者覆盖旧值，适合Piper只保留最新反馈。
- `FrameBuffer`：默认最多120帧；`nearest(timestamp, tolerance)`按主机单调时钟选择最近帧，用于软件同步。

### 7.3 `common/event_log.py`

每个episode创建：

```text
deploy/runs/<episode_id>/events.jsonl
```

每条记录自动加入：

- `event`
- Unix wall time
- `monotonic_ns`

numpy、dataclass和Path自动转换为JSON。每次写入立即flush，便于崩溃后保留日志。

### 7.4 相机适配

`CameraDevice`统一线程生命周期和异常捕获。

Orbbec：

- 枚举设备并按可选serial选择。
- 请求RGB分辨率和FPS，失败时退回默认profile。
- 支持I420/MJPG/YUYV/NV21/NV12/UYVY转RGB888。
- 输出`H×W×3 uint8 RGB`。

RealSense：

- 可选serial。
- 明确请求`rs.format.rgb8`。
- 当前只启用color stream，没有深度、IMU和对齐。

### 7.5 `devices/piper_robot.py`

启动：

```python
C_PiperInterface_V2(
    can_name=...,
    judge_flag=...,
    can_auto_init=True,
    dh_is_offset=...,
    start_sdk_joint_limit=True,
    start_sdk_gripper_limit=True,
)
ConnectPort(piper_init=False, start_thread=True)
```

反馈线程约每5 ms读取：

- `GetArmJointMsgs()`
- `GetArmEndPoseMsgs()`
- `GetArmGripperMsgs()`

单位转换：

- 关节：0.001° -> rad。
- XYZ：0.001 mm -> m，即除以1,000,000。
- RPY：0.001° -> rad。
- 夹爪行程：0.001 mm -> m。

SDK EndPose经`PiperFrameTransform.sdk_to_model_pose()`后发布为`RobotState`。

运动只有显式调用`enable_motion()`后才允许：

```python
EnableArm(7)
```

动作下发：

1. 验证7维。
2. model TCP逆转换到SDK link6。
3. 米/rad换为SDK整数单位。
4. 调用`MotionCtrl_2(0x01, 0x00, speed_percent, 0x00)`。
5. 调用`EndPoseCtrl(...)`。
6. 仅当`command_gripper=true`时调用`GripperCtrl`。

停止时如果曾使能，则终止当前轨迹并用MOVE J锁定反馈关节位置，保持电机使能后再断开SDK连接。实机已经确认 `DisableArm(7)`会移除重力支撑并导致机械臂砸落，因此禁止将自动DisableArm作为正常清理动作。只有在机械臂已经被可靠机械支撑时才可人工失能或断电。

### 7.6 `policy/observation_builder.py`

- 每个控制tick以当前主机单调时间为同步目标。
- 每路相机必须在`camera_sync_tolerance_ms`内找到最近帧，否则记录`observation_skipped`。
- 保存历史sample，按`history_indices=[-2,0]`选帧。
- 历史不足时索引夹到第一个sample，因此启动初期会重复早期帧。
- 图像堆叠为`[T,H,W,C]`。
- state堆叠为`[T,D]`。

### 7.7 `policy/inference_worker.py`

- 输入队列maxsize=1：模型忙时新观测覆盖尚未处理的旧观测。
- 输出队列maxsize=2：主循环总是取出最新可用结果。
- 在线程中加载checkpoint，避免阻塞设备主循环。
- 根据checkpoint真实input shape调整图片到360×480。
- `permute`后显式`contiguous()`，修复上游`view()`对非连续tensor报错。
- 根据checkpoint state维度裁剪/补零，修复6维state和7维本地RobotState不匹配。
- 调用`predict_action_chunk()`。
- 若`use_delta_action`，对前`action_dim-1`维加最新raw state，夹爪不加。
- 输出带来源观测index和时间戳的`ActionChunk`。

### 7.8 `policy/action_scheduler.py`

这是对上游LAAS队列的本地重写，核心规则：

1. 拒绝错误episode。
2. 拒绝来源index倒退或重复的chunk。
3. 拒绝观测年龄超过`max_action_age_ms`的chunk。
4. `skip = current_index - observation_index`。
5. 丢弃chunk前`skip`个已过期动作。
6. 若整个chunk均过期，拒绝整个chunk。
7. 新chunk从其有效起点开始替换尚未执行的未来动作。
8. 不覆盖已经执行的index。

统计：

- accepted_chunks
- stale_chunks
- expired_actions
- executed_actions

### 7.9 `control/safety_filter.py`

输入必须是绝对model TCP动作`[7]`。当前顺序：

1. shape检查。
2. NaN/Inf检查。
3. **先检查原始目标**是否在model TCP workspace内。
4. 计算当前位置到目标的平移，限制到`max_translation_step_m`。
5. 再检查限幅后的目标；当前位置在workspace外时不能借小步逐渐进入。
6. Euler差值包装到`[-π,π)`，限制三分量差向量范数。
7. 夹爪按阈值映射到物理开口。

先检查原始目标是故意的：不能把明显错误的模型目标裁剪到边界后持续驱动机械臂撞向安全边界。

### 7.10 `run.py`

启动顺序：

1. 加载并校验YAML。
2. 创建episode日志。
3. 启动Piper反馈。
4. 启动双相机。
5. 启动模型线程。
6. 等待双相机首帧、Piper首个state和模型ready。
7. 重置scheduler。
8. execute时进行双重确认并使能机械臂。
9. 进入25 Hz控制循环。

每个tick：

1. 检查所有worker错误。
2. 检查Piper反馈是否超时。
3. 尝试构造并提交最新观测。
4. 获取最新ActionChunk并提交scheduler。
5. pop当前index动作。
6. 运行SafetyFilter。
7. 写日志。
8. shadow只记录；execute才下发。
9. 检查动作流是否过期。

退出始终在`finally`中停止模型、相机、Piper并写`runtime_stop`。

---

## 8. 配置完整说明

当前配置：`deploy/configs/piper_gemini_d435i.yaml`

### model

- `checkpoint`：必须是包含config.json和model.safetensors的目录。
- `device`：通常cuda。
- `task`：语言指令，必须和场景语义匹配。
- `rotation`：当前适配器仅支持euler。

### cameras

每路：

- `driver`：orbbec或realsense。
- `enabled`。
- `serial`：空表示选择首个匹配设备；多相机环境必须填写。
- `width/height/fps`：当前640×480@30采集，worker缩放到checkpoint要求的480×360。

### robot

- `can_interface: can0`
- `dh_is_offset: 1`：SDK本地FK的J1-J2 2° DH偏置，必须和硬件/SDK版本核对。
- `official_can_adapter`
- `auto_enable`：默认false。
- `command_gripper`：默认false。
- `command_speed_percent: 10`
- `sdk_to_model_translation_m`
- `sdk_to_model_euler_xyz_rad`

### runtime

- `mode`：shadow或execute。
- `control_hz: 25`
- `camera_sync_tolerance_ms: 50`
- `sensor_timeout_ms: 250`
- `startup_timeout_s: 300`
- `output_dir`
- `history_indices: [-2,0]`

### safety

- `workspace_min_m/max_m`：**model TCP/base_link坐标**，当前仍是保守占位边界，需要基于桌面和碰撞实测。
- `max_translation_step_m: 0.015`
- `max_rotation_step_rad: 0.12`
- `max_action_age_ms: 800`
- `stale_action_hold_ms: 200`
- `hold_on_stale_action: true` in `piper_sequential.yaml`：execute动作流断档时记录`action_stream_stale_hold`并保持最后JointCtrl目标，不因单次模型推理抖动立即退出。其他配置默认仍可保持旧的断流停机语义。
- `gripper_open_threshold: 0`
- `gripper_min_m: 0`
- `gripper_max_m: 0.07`

---

## 9. shadow与execute语义（基础设计，执行后端以第27节为准）

### shadow

- 读取全部真实设备。
- 加载真实模型并推理。
- 运行LAAS和SafetyFilter。
- 写完整动作和拒绝日志。
- 不调用`EnableArm`、`EndPoseCtrl`或`GripperCtrl`。

### execute

同时要求：

```text
runtime.mode == execute
robot.auto_enable == true
命令行存在 --confirm-motion
```

缺一项立即拒绝启动。execute中任何SafetyViolation会抛出并退出，Piper在finally中终止轨迹并保持当前关节位置，不自动失能。

**这是早期阶段的禁用结论。当前execute已用于受控实验，但仍非生产安全控制器；必须使用当前三重门控、主机IK联锁和现场监护，不得绕过配置与状态检查。**

---

## 10. 日志事件

主要事件：

| event | 含义 |
|---|---|
| runtime_start | 包含完整展开配置 |
| devices_ready | 双相机、Piper、模型均ready |
| observation_skipped | 相机同步失败等 |
| action_chunk | chunk来源、推理时间、是否被scheduler接受、完整动作 |
| action | raw/safe动作、state、executed |
| safety_reject | 拒绝原因 |
| motion_enabled | execute实际使能 |
| control_overrun | 主循环超过周期 |
| runtime_stop | scheduler最终统计 |

shadow中`action.executed`必须为false。

汇总：

```bash
RUN=$(ls -td deploy/runs/* | head -1)
python -m deploy.tools.summarize_log "$RUN/events.jsonl"
```

---

## 11. 早期阶段已验证结果（历史记录）

### 11.1 已完成

- 双相机SDK握手成功。
- Piper CAN握手和反馈成功。
- 模型checkpoint成功加载。
- 6维state适配已修复。
- 图像contiguous问题已修复。
- shadow长期产生`action_chunk`和`action`事件。
- LAAS能按延迟跳过动作。
- EndPose与FK joint6坐标关系已实机确认。
- model TCP双向变换有round-trip单测。
- 当时离线测试：13 passed；当前最新记录见第0、26和27节。

### 11.2 坐标验证

`verify_piper_frames.py`结果：

- 100个有效样本。
- EndPose匹配FK joint6。
- arm status全部normal。
- 当前固定变换可以保留。

### 11.3 最新shadow

```text
action_chunk: 1151
action: 32
safety_reject: 8278
inference mean: 0.2825 s
inference p95: 0.3161 s
LAAS skipped mean: 8.25
```

8278次拒绝全部为：

```text
Target position outside workspace
```

示例非法绝对目标：

```text
[-0.000356, 0.057599, 0.086029]
```

当前位置：

```text
[0.170183, -0.011534, 0.114729]
```

因此该例对应约`Δx=-0.1705 m`，属于明显异常模型输出。安全层行为正确。

---

## 12. 当前核心问题：模型分布外，而不是坐标变换失败

当前model TCP与训练统计的标准化偏差：

| 分量 | 当前 | 训练mean | std | z-score |
|---|---:|---:|---:|---:|
| X | 0.170 | 0.385 | 0.125 | -1.72 |
| Y | -0.012 | -0.001 | 0.152 | -0.07 |
| Z | 0.115 | 0.172 | 0.106 | -0.54 |
| RX | 3.262 | 3.136 | 0.059 | +2.13 |
| RY | -1.166 | -0.118 | 0.240 | **-4.37** |
| RZ | 6.082 | 2.910 | 2.819 | +1.13 |

主要异常是RY：训练中心约-6.8°，当前约-66.8°。

仿真配置初始TCP：

```text
position ≈ [0.373, 0, 0.271] m
quaternion WXYZ = [0, 0.9739, 0, 0.227]
Euler(model convention) ≈ [π, -0.458, 0]
```

仿真初始关节：

```text
[0°, 90°, -90°, 0°, 68.8°, 0°]
```

当前实机关节：

```text
[-4.7°, 0°, 0°, 1.3°, 28.8°, 1.5°]
```

当前机械臂明显处于折叠构型。模型不输入关节角，因此无法感知肘部构型、IK分支、关节极限或奇异点接近程度。

可能同时存在的域差异：

- 当前TCP位置/姿态远离训练起始分布。
- 关节构型远离仿真。
- 真机相机外参、焦距、色彩、遮挡与仿真不同。
- 场景和任务文本可能不匹配DOM checkpoint。
- 仿真到真机视觉域差异。
- 当前公开checkpoint是否包含当前真实Piper embodiment的post-training数据尚未确认。

禁止通过降低`workspace_min.x`来掩盖这个问题。负X或接近base的目标可能导致底座/桌面碰撞、IK无解和构型跳变。

---

## 13. 夹爪完整路径

反馈：

```text
GetArmGripperMsgs().gripper_state.grippers_angle
0.001 mm -> meter
```

当前checkpoint state为6维，因此夹爪反馈被裁掉，不参与策略输入。

模型输出第7维为连续数值，安全层映射：

```text
value >= gripper_open_threshold -> gripper_max_m
value <  gripper_open_threshold -> gripper_min_m
```

下发：

```text
meters × 1,000,000 -> SDK 0.001 mm整数
GripperCtrl(gripper_units, 1000, 0x01, 0)
```

当前`command_gripper=false`，不会实际动作。需要现场验证：

- 实体最大安全开口是否确实为0.07 m。
- 开闭方向。
- 夹爪归零状态。
- 力矩1000（1 N·m）是否合适。
- 是否需要迟滞，避免输出在阈值附近抖动。
- 是否需要连续开口映射而不是二值。

---

## 14. 机械臂构型、IK与奇异点（旧MOVE P阶段记录）

仿真使用Isaac DifferentialIKController：

- command type：absolute pose。
- IK method：DLS。
- 控制body：gripper_base。
- 工具偏移约0.137 m。

真机Python部署没有运行Isaac DLS，也没有自行实现IK。它将link6笛卡尔目标通过`EndPoseCtrl`发送到Piper控制器，由固件完成运动学/轨迹处理。

Piper官方状态包含：

- `0x02`：No solution。
- `0x03`：Singularity。
- `0x04`：Target angle exceeds limit。
- 以及急停、碰撞、关节通信、刹车等状态。

**该缺口已在后续阶段补齐：当前`RobotState`和runtime已读取并联锁ArmStatus、控制模式和关节限位标志。当前主机IK和执行状态以第26—27节为准。**

此外仅靠固件报错属于事后/临界检测；推荐增加：

- SDK FK或独立URDF FK。
- Jacobian最小奇异值/条件数。
- 关节软限位余量。
- 关节速度和加速度。
- IK解连续性和构型分支跳变检测。
- 机器人本体、桌面、相机和夹爪碰撞模型。

---

## 15. 早期安全系统的能力和边界（历史基线）

### 已实现

- shadow默认无运动。
- execute三重条件（mode、auto_enable、CLI确认）。
- 设备/推理线程错误watchdog。
- Piper feedback超时。
- 相机软件同步失败跳过观测。
- 动作年龄和流中断检查。
- NaN/Inf和shape检查。
- model TCP目标workspace。
- 单步平移限制。
- 单步Euler差限制。
- 夹爪独立开关。
- 日志可追溯raw/safe/state。
- 停止时终止轨迹并以当前关节目标保持使能，避免重力掉落。

### 当时未实现（其中ArmStatus和主机侧关节限位IK后续已实现）

- ArmStatus实时联锁（后续已实现，保留此项仅作历史记录）。
- 急停输入的软件状态确认。
- 预执行IK可达性。
- 奇异点预测。
- 关节软限位/速度/加速度独立检查。
- 笛卡尔速度和加速度严格约束。
- 控制器实际跟踪误差。
- 路径中间点碰撞。
- 自碰撞/环境碰撞。
- 相机外壳和线缆碰撞。
- raw delta异常阈值（当前只检查最终目标和限步）。
- 夹爪力/物体检测。
- 多进程心跳或外部安全PLC。

workspace只是model TCP长方体，不等于机器人所有连杆和路径安全。

---

## 16. 原版与当前实现差异

| 方面 | 原版/论文 | 当前deploy |
|---|---|---|
| 环境 | Isaac Lab或论文未公开真机系统 | Gemini 305、D435i、Piper |
| IK | Isaac DLS | Piper固件笛卡尔控制 |
| 观测传输 | ZeroMQ/pickle和仿真状态 | 进程内dataclass、线程和时间戳 |
| Continuous Inference | 推理执行重叠 | 单槽异步推理线程 |
| LAAS | 根据延迟跳过旧动作 | episode/index/单调时间戳scheduler |
| 对象状态 | 仿真GT或EfficientTAM real-world simulator | 未实现 |
| 相机 | 论文通常三视角 | 当前双RGB |
| 安全 | 仿真终止条件 | 初步真机笛卡尔安全层 |
| 日志 | 仿真pickle/结果 | JSONL逐事件 |
| TCP | Isaac gripper_base+offset | 显式SDK link6↔model TCP |
| 夹爪 | 仿真±1 | 阈值映射为Piper行程 |

---

## 17. 命令与验证顺序（历史基线，运动前须核对当前CLI与YAML）

### 17.1 仅设备诊断

```bash
python -m deploy.diagnose \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --seconds 10
```

不加载模型、不使能、不发动作。

### 17.2 Piper坐标只读验证

```bash
python -m deploy.tools.verify_piper_frames \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --seconds 10
```

该工具连接CAN、启用SDK本地FK并只读比较EndPose/FK。不要和其他Piper进程并行运行。

### 17.3 shadow

```bash
python -m deploy.run \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --mode shadow
```

### 17.3a 低速回训练初始构型（独立工具）

必须先dry-run：

```bash
python -m deploy.tools.move_to_training_start \
  --config deploy/configs/piper_gemini_d435i.yaml
```

实际运动还需要 `--execute --confirm-motion`和交互确认，默认速度5%、硬上限10%。它使用MOVE J到原版初始关节，不控制夹爪，并监控ArmStatus、关节限位、超时和到位误差。该工具不具备碰撞规划，现场必须先人工确认完整扫掠空间。

### 17.4 测试

```bash
python -m pytest -q deploy/tests
python -m compileall -q deploy
```

当时预期：13 passed。当前文档最新记录为26 passed，必须以本机实际运行结果为准。

### 17.5 execute

以下是早期预留的execute命令形态。当前已进入受控execute验收阶段，但运动前必须确认当前后端为`host_ik_move_j`、状态联锁正常、配置参数未被放宽，并完成现场监护：

```bash
python -m deploy.run \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --mode execute \
  --confirm-motion
```

并且YAML还必须显式设置`auto_enable: true`。

首次真机执行使用独立的 `deploy/configs/piper_gemini_d435i_first_execute.yaml`。首轮最低档完成后，当前配置已调整为Piper速度3%，每个25 Hz控制周期最多推进3 mm和0.02 rad，夹爪关闭，15秒后自动停止动作更新并保持反馈关节位置。CLI仍要求 `--confirm-motion`。

---

## 18. 早期测试覆盖（历史基线）

### `test_action_scheduler.py`

- 丢弃chunk过期前缀。
- 新chunk替换未来动作。
- 拒绝错误episode和旧chunk。
- 拒绝超过年龄的chunk。

### `test_inference_worker.py`

- 7维本地state裁剪到checkpoint 6维。
- 低维state补零。

### `test_piper_frames.py`

- SDK单位姿映射到Rz(π)+Z 0.1334模型TCP。
- 任意姿态的SDK→模型→SDK round trip。

### `test_safety_filter.py`

- 平移/旋转单步限幅。
- workspace拒绝。
- NaN拒绝。
- 夹爪二值映射。
- 当前state在workspace外时拒绝通过限步逐渐进入。

### 尚需新增测试

- ArmStatus联锁。
- Piper命令转换使用mock断言实际SDK整数值。
- Euler奇异附近的变换和限幅。
- raw delta异常拒绝。
- 相机时间戳抖动和断流。
- runtime错误时终止轨迹、锁定当前关节并保持使能。
- execute三重门控。
- checkpoint加载集成测试。
- 记录回放和离线策略回归测试。

---

## 19. 早期二次开发推荐顺序（历史计划）

### P0：execute前强制完成

1. 在Piper反馈线程读取`GetArmStatus()`，扩展`RobotState`或单独状态消息。
2. 主循环对所有非normal状态fail closed；重点处理急停、无解、奇异、越界、碰撞、通信和刹车。
3. 增加关节角软限位和与仿真起始构型的偏差检查。
4. 增加raw模型delta上限，明显异常目标直接拒绝整个chunk。
5. 明确笛卡尔目标控制模式和Piper固件版本；验证25 Hz重复`EndPoseCtrl`的控制语义。
6. 建立实体workspace、桌面平面和腕部相机碰撞边界。
7. 以厂商示教/受控方式将机械臂放到接近训练起始构型，只做shadow。

### P1：模型可用性

1. 记录双相机实际图像并和仿真训练视角并排检查。
2. 固定camera serial，避免设备枚举顺序变化。
3. 建立state z-score和action delta统计工具。
4. 对每个chunk记录各动作越界维度和随horizon变化。
5. 确认checkpoint任务、物体、容器和语言提示。
6. 若仍严重分布外，采集当前Piper真机数据做post-training/finetune，而不是放宽安全层。

### P2：控制质量

1. 增加跟踪误差监控：commanded model TCP vs feedback model TCP。
2. 将单步限制升级为基于真实dt的速度/加速度限制。
3. 研究Piper MOVE P/MOVE L/MOVE CPV哪种最适合25 Hz流式目标。
4. 增加姿态的四元数/旋转向量安全限幅，保持模型Euler接口只在边界转换。
5. 夹爪增加迟滞、力矩限制和状态确认。

### P3：论文真机系统复现

1. 增加第三视角相机（若目标是完整复现论文）。
2. EfficientTAM或替代分割。
3. 多视角三角化和标定。
4. 对象6D位姿/线速度/角速度。
5. 自动数据收集状态机。
6. 真实Piper embodiment后训练。

---

## 20. 必须保持的工程不变量

后续AI修改代码时不得破坏：

1. **默认shadow，默认auto_enable=false，默认command_gripper=false。**
2. 构造和start设备不得产生运动。
3. SDK坐标与模型坐标只在`PiperFrameTransform`边界转换一次。
4. 安全workspace始终定义在`base_link -> model_tcp`。
5. 不得将非法原始目标简单clamp后执行。
6. 推理线程不得阻塞25 Hz主循环。
7. 新chunk不得覆盖已经执行的动作index。
8. 所有时间有效性使用`time.monotonic_ns()`，不能用wall clock做超时。
9. 任意设备、模型或安全异常都必须终止运动并保持当前位置；Piper不能用自动DisableArm实现fail closed，因为失能会在重力下掉落。
10. 日志必须区分raw、safe和executed。
11. 不得因为shadow大量reject而降低workspace来“让日志通过”。
12. 未完成ArmStatus、关节和碰撞联锁前不得宣布execute安全。

---

## 21. 已知文档/API注意事项

- Piper官方文档将`GetArmEndPoseMsgs`称为end effector，但没有明确命名link6；本项目已通过与`GetFK()[-1]`实测比较确认当前固件行为。
- 官方接口文档对`GetFK`和夹爪反馈单位存在表述不完全一致；本地piper-sdk 0.6.1源码和实测量级显示FK返回mm/degree浮点，EndPose/Gripper CAN反馈是0.001 mm整数。
- Orbbec导入时会输出extensions加载路径，不能用简单命令替换捕获`pyorbbecsdk.__file__`的stdout，因为该提示会污染路径。
- Hugging Face基础SmolLM2配置可能首次联网下载；部署机器应预缓存并设置HF_TOKEN或离线缓存。
- 模型加载日志中的config override来自checkpoint与默认policy配置合并，不一定是错误，但每次更换checkpoint必须重新核对input/output、history、chunk和delta配置。

---

## 22. 接手者首次检查清单

1. 阅读本文、`deploy/README.md`、`deploy/DEBUGGING.md`。
2. 检查`git status`，当前`deploy/`可能尚未提交，不得覆盖用户修改。
3. 运行13个离线测试。
4. 确认YAML仍为shadow/auto_enable false/command_gripper false。
5. 确认相机角色没有对调。
6. 运行diagnose。
7. 运行`verify_piper_frames`，确认EndPose仍匹配FK joint6。
8. 查看最新events.jsonl，而不是仅看终端输出。
9. 统计state z-score、raw delta和每个维度的workspace拒绝。
10. 在所有P0安全项完成前只允许只读和shadow开发。

---

## 23. 关键源码索引

上游/原版：

- `simulations/simulate.py`：世界到机器人根坐标、state/action组织。
- `simulations/helpers.py`：相对位置和方向定义。
- `simulations/configs/robot_cfg.py`：DLS IK、gripper_base和TCP偏移。
- `simulations/robots/PIPER/piper_description.urdf`：link6到gripper_base固定Rz(π)。
- `simulations/configs/sim_cfg.yaml`：Piper仿真初始TCP。
- `simulations/robots/piper.py`：仿真初始关节。
- `utils/helpers.py`：四元数WXYZ和Euler XYZ转换。
- `utils/datasets.py`：delta action训练转换。
- `scripts/inference.py`：原版推理和delta恢复。
- `policies/dynamicvla/modeling_dynamicvla.py`：模型chunk推理。

真机：

- `deploy/run.py`
- `deploy/config.py`
- `deploy/devices/piper_robot.py`
- `deploy/devices/piper_frames.py`
- `deploy/policy/inference_worker.py`
- `deploy/policy/action_scheduler.py`
- `deploy/control/safety_filter.py`
- `deploy/tools/verify_piper_frames.py`

---

## 24. 早期阶段最终判断（历史记录）

截至该阶段：

1. 相机、CAN、模型、异步推理和日志链路可用。
2. Piper EndPose/link6与DynamicVLA model TCP的坐标关系已经通过代码、URDF和实机数据交叉确认。
3. LAAS按设计工作，实测会跳过约8个过期动作点。
4. 当前模型TCP姿态尤其RY严重偏离训练分布。
5. 最新shadow约99.6%的调度动作因模型绝对目标越出workspace而被拒绝。
6. 这不是降低安全边界可以解决的问题；需要先对齐起始TCP、关节构型、相机视角和任务场景，必要时进行真机post-training。
7. 当时安全层不足以支持execute；其中ArmStatus联锁和主机侧关节限位感知IK后续已实现，碰撞、轨迹和严格跟踪联锁仍未完成。

下一位开发者应从“P0安全联锁 + 分布诊断工具”继续，而不是直接开放运动。

---

## 25. 2026-06-24 MOVE P实机执行阶段（历史记录）

> 本节记录2026-06-23初版之后的MOVE P真机阶段，用于保留握手、0x04、回位、ArmStatus和首chunk bootstrap的问题历史。本节不再是当前执行后端的权威说明；若与第0、26、27节或当前源码冲突，以后者为准。

### 25.1 当时总状态

当时链路已经实际完成：

```text
Gemini 305 wrist_cam + D435i opst_cam
→ observation builder
→ DynamicVLA异步推理
→ delta action还原为绝对model TCP目标
→ LAAS scheduler
→ SafetyFilter
→ model TCP到Piper SDK link6坐标变换
→ MotionCtrl_2(MOVE P) + EndPoseCtrl
→ Piper真实反馈和JSONL日志
```

已经发生过真实MOVE P运动、夹爪命令和正常MOVE J自动回位，但系统仍缺少：

- 下发前关节限位感知IK。
- 奇异值/构型分支预测。
- 连杆、桌面、相机和线缆碰撞规划。
- 严格跟踪误差、速度和加速度联锁。
- 真机示范采集器和针对当前相机域的post-training。

因此execute只能作为有人监护的实验模式，不能视为生产安全控制器。

当前离线测试：

```bash
python -m pytest -q deploy/tests
# 22 passed
```

### 25.2 当前关键文件与新增模块

新增或重点修改：

- `deploy/targets.py`
  - 统一定义实机标定待机/起始关节。
  - 避免`deploy.run`和独立回位工具维护两份不一致常量。
- `deploy/tools/diagnose_piper_cartesian.py`
  - 默认只读。
  - 显式确认后以1 mm MOVE P验证EndPoseCtrl链路。
  - 结束后MOVE J保持当前关节，不失能。
- `deploy/tools/move_to_training_start.py`
  - 低速MOVE J到统一待机点。
  - FK采样检查model TCP工作空间。
  - 监控ArmStatus、关节反馈、超时和到位稳定性。
  - 回位开始和完成时夹爪闭合到0 mm。
- `deploy/devices/piper_robot.py`
  - 完整六轴EnablePiper握手。
  - MOVE P前预装当前EndPose，避免旧笛卡尔目标被激活。
  - CAN/MOVE P模式反馈校验。
  - 正常结束MOVE J自动回位。
  - 0x04回位前通过当前关节MOVE J保持进行已知错误恢复。
- `deploy/common/messages.py`
  - `RobotState`增加`ctrl_mode`、`arm_status`、`mode_feed`、`motion_status`、`err_code`和六轴超限位标志。
- `deploy/policy/action_scheduler.py`
  - 首chunk bootstrap重新锚定。
  - 后续chunk保留正常LAAS过期跳步。
- `deploy/run.py`
  - 实时Piper状态联锁。
  - 正常结束自动回位。
  - 新增状态故障和回位日志事件。

### 25.3 当前专用execute配置

文件：`deploy/configs/piper_gemini_d435i_first_execute.yaml`

当前关键值：

```yaml
robot:
  auto_enable: true
  command_gripper: true
  command_speed_percent: 5

runtime:
  mode: execute
  control_hz: 25.0
  max_execute_seconds: 30
  return_to_training_start_on_normal_exit: true
  return_speed_percent: 5
  return_timeout_s: 45

safety:
  max_translation_step_m: 0.01
  max_rotation_step_rad: 0.08
  gripper_open_threshold: 0.0
  gripper_min_m: 0.0
  gripper_max_m: 0.07
```

基础配置`piper_gemini_d435i.yaml`仍用于shadow。不要将专用execute配置视作默认安全配置。

### 25.4 新实机标定起始/待机点

用户最终指定：

```text
joint degrees = [0.0, 89.913, -80.913, 0.0, 58.398, 0.0]
```

统一定义于`deploy/targets.py`：

```text
TRAINING_START_DEG
TRAINING_START_RAD
```

Piper SDK FK（dh_is_offset=1）约为：

```text
SDK link6 position = [331.991, 0.000, 329.257] mm
SDK link6 Euler    = [180.000, 27.602, 180.000] deg
model TCP position = [0.393799, 0.000000, 0.211039] m
model Euler XYZ    = [pi, -0.481746, 0] rad
```

对应仿真四元数WXYZ约为：

```text
[0.0, 0.971130, 0.0, 0.238550]
```

J5从旧起点约68.8°降低到58.398°，相对URDF +69.9°上限多出约11.5°余量。

重要：当前这个目标只控制独立回位和正常结束自动回位。`deploy.run`不会在模型启动前自动移动到该点；启动前必须人工确认或先运行回位工具。

回位命令：

```bash
python -m deploy.tools.move_to_training_start \
  --config deploy/configs/piper_gemini_d435i_first_execute.yaml \
  --speed-percent 5 \
  --execute --confirm-motion
```

交互确认文本：

```text
MOVE_PIPER_SLOWLY_TO_TRAINING_START
```

回位开始和完成时均发送：

```python
GripperCtrl(0, 1000, 0x01, 0x00)
```

即夹爪0 mm闭合、1 N·m命令力矩。

### 25.5 MOVE P握手和残留目标问题

早期表现：Python记录`executed=true`，模型和安全层给出明显不同目标，但15秒内Piper末端反馈完全不变。

独立1 mm诊断确认：

1. 六轴使能正常。
2. `ctrl_mode=CAN`正常。
3. 直接切MOVE P时会在尚未发送新1 mm目标前出现`0x04 target_joint_limit`。
4. 原因是Piper内部保留了上一进程的笛卡尔目标；切MOVE J到MOVE P时旧目标被立即激活。
5. 在MOVE J状态先用当前实测EndPose预装目标，再切MOVE P，状态保持normal且1 mm命令产生了可测位移。

当前`enable_motion()`原则：

```text
EnablePiper循环直到六轴全部使能
→ 读取当前EndPose
→ EndPoseCtrl(当前位姿)覆盖旧寄存器
→ MotionCtrl_2(CAN, MOVE_P, speed, position-speed)
→ 再发当前EndPose
→ 只在ctrl_mode=1、mode_feed=0、arm_status=0后允许模型动作
```

实际还观察到控制器保持`ctrl_mode=0x00 standby`、单次0x151没有进入CAN的情况。代码已增加3秒重复握手。Piper官方示例也是周期性发送`MotionCtrl_2 + EndPoseCtrl`，不是把0x151视为一次必达事务。

不要随意调用：

```python
MotionCtrl_1(0x02, 0, 0)
```

SDK把它描述为恢复/复位；部分状态下可能导致失能和机械臂下落。只有机械臂得到物理支撑并明确理解固件行为时才能试验。

### 25.6 停止和保持策略

已确认`DisableArm`/`DisablePiper`会移除电机支撑，机械臂可能在重力下砸落。因此正常清理和异常清理均不自动失能。

当前策略：

```text
读取当前六关节反馈
→ MotionCtrl_2(CAN, MOVE_J, low speed)
→ JointCtrl(当前反馈关节)
→ 保持电机使能
→ DisconnectPort
```

正常时间上限退出：

```text
execute_time_limit
→ return_to_training_start_begin
→ 慢速MOVE J到TRAINING_START_DEG
→ return_to_training_start_complete
→ 当前关节保持
→ runtime_stop
```

异常、Ctrl+C、通信故障或模型故障：

```text
不自动回远处起始点
→ 保持错误发生时的当前关节
→ runtime_stop
```

这是有意设计：故障状态下不得自动开始新的长路径运动。

### 25.7 ArmStatus实时联锁

`RobotState`当前携带：

```text
ctrl_mode
arm_status
mode_feed
motion_status
err_code
joint_limit_flags[6]
```

execute控制循环要求：

```text
ctrl_mode == 0x01  # CAN
mode_feed == 0x00  # MOVE P
arm_status == 0x00 # normal
```

否则写入：

```text
robot_status_fault
```

事件包含：

```text
index
ctrl_mode/mode_feed/arm_status/motion_status
err_code
joint_limit_flags
limit_joints
joint_degrees
model TCP state
```

然后立即停止模型动作并进入当前位置MOVE J保持。

### 25.8 0x04根因和J5离线确认

Piper状态：

```text
0x04 = target joint angle exceeds limit
```

它表示Piper内部对`EndPoseCtrl`做IK后，目标关节超限；不一定表示当前反馈关节已经越限。

关键日志：

```text
deploy/runs/20260624-190923-b9525d/events.jsonl
```

故障发生在第一条模型动作之后：

```text
current model TCP = [0.385111, 0.000000, 0.247504,
                     pi, -0.455095, 0]
safe target       = [0.383875, 0.006360, 0.239886,
                     3.141775, -0.375095, 0.000178]
```

反馈：

```text
arm_status = 0x04
err_code = 0
joint_limit_flags = [false, false, false, false, false, false]
current joints = [0.022, 89.998, -89.977, 0, 68.902, 0] deg
```

固件对“被拒绝的IK目标”没有设置逐轴err bits，因此仅靠CAN反馈不能知道目标哪一轴越界。使用本地Piper FK、项目URDF和SciPy从当前关节对该安全目标做离线数值IK，得到：

```text
unbounded IK J5 target ≈ 74.001°
URDF J5 upper          ≈ 69.901°
overshoot              ≈ 4.10°
```

其余目标关节在限位内，因此这条已记录目标可确认是J5导致0x04。受限IK会卡在J5上限且保留明显位姿残差，说明该完整6D目标在当前构型和限位下不可达。

当前仍未实现在线IK预检查；runtime只能在固件拒绝后的下一个控制tick fail closed。

### 25.9 真实运动日志结论

`20260624-051911-aa543a`：

- 353条动作实际下发。
- 无`safety_reject`。
- 15秒正常退出。
- model TCP净位移约`[-0.89, +4.66, -8.49] mm`，合位移约9.72 mm。
- 累计轨迹约17.34 mm。
- 说明坐标变换、MOVE P和反馈链路已经真实打通。

`20260624-053432-20ab35`：

- 30秒执行，Piper主要在前约7秒运动，之后长期不动。
- 正常回位事件完整：
  - `return_to_training_start_begin`
  - `return_to_training_start_complete`
- 旧待机目标到达反馈约`[0.0, 89.882, -89.964, 0.0, 68.723, 0.0]°`。
- 后续ArmStatus联锁证明此前“先动后长期不动”是中途0x04未被旧runtime观察，而不是Python没有继续调用SDK。

`20260624-060502-a50c37`和`20260624-190923-b9525d`：

- 第一条动作后约40 ms即收到0x04。
- 状态联锁按设计立即停止。
- 夹爪不是0x04原因；它与MOVE P使用不同控制接口。

### 25.10 LAAS首chunk bootstrap修复

原调度逻辑按照：

```text
skip = current_control_index - chunk_observation_index
```

首次真实推理约0.7秒，25 Hz下会跳过约18步。20步chunk只剩最后2步，第一条真机动作实际是`action[18]`。

稳定运行阶段这个逻辑合理，因为推理新chunk期间旧chunk仍在执行；被跳过的时间由旧动作填充。首次推理没有旧chunk，机械臂一直保持原位，跳过前18步等于错误假设机器人已经走完这段轨迹。

当前修复：

```text
first accepted chunk:
  skip = 0
  action[0]重新锚定到当前control tick

second and later chunks:
  恢复正常LAAS过期跳步
```

`SchedulerStats`新增：

```text
bootstrap_chunks
```

正常episode应为1。该修复不关闭Continuous Inference，也不改变稳定阶段LAAS。

### 25.11 模型起点、绝对动作和训练分布

Checkpoint：

```text
use_delta_action = true
```

模型输出前六维是delta，`DynamicVLAWorker`会加上最新6D model TCP state，因此安全层看到绝对目标。模型不直接输出Piper关节角。

原仿真Piper有两层起点：

```text
reset joints ≈ [0, 90, -90, 0, 68.75, 0] deg
state-machine INIT model TCP = [0.373, 0.0, 0.271] m
INIT quaternion WXYZ = [0, 0.9739, 0, 0.227]
```

Pick状态机INIT持续约0.48秒，并反复命令固定init_pose。模型训练数据因此包含固定起点和固定开场轨迹。

仅修改`deploy/targets.py`不会修改checkpoint训练分布。新实机起点对原checkpoint属于分布变化；模型可能继续预测旧开场轨迹或接近物体轨迹。首次LAAS跳步曾进一步放大这种偏差。

### 25.12 夹爪当前实现

当前first-execute配置已开启：

```yaml
command_gripper: true
```

映射：

```text
model gripper >= 0 → 0.07 m张开
model gripper <  0 → 0.00 m闭合
```

SDK：

```python
GripperCtrl(gripper_units, 1000, 0x01, 0)
```

- 行程单位转换：meter × 1,000,000 → 0.001 mm整数。
- 力矩命令：1000，即1 N·m。
- 模型输入仍只有6维末端位姿，夹爪反馈不进入checkpoint。
- 当前没有迟滞和稳定时间；若模型符号频繁切换，夹爪可能反复开闭，仍需增加去抖。
- 异常退出不会自动闭合夹爪；夹爪保持最后一次命令。正常回位和独立回位会闭合。

### 25.13 真机相机角色

最终映射保持：

```text
Gemini 305 → wrist_cam → 腕部视角
D435i      → opst_cam  → 固定第三视角
```

任何新数据采集和微调都必须保持同样feature key、视角语义和图像顺序。仅交换设备驱动但不交换模型key会造成严重分布错误。

### 25.14 针对新起点的微调路线

要让checkpoint真正适应新起点，至少需要修改仿真数字孪生并重新生成数据：

1. `simulations/robots/piper.py`
   - reset关节改为：

```text
[0, 1.569278, -1.412198, 0, 1.019237, 0] rad
```

2. `simulations/configs/sim_cfg.yaml`
   - Piper init pose改为：

```yaml
init_pose: [0.393799, 0.0, 0.211039,
            0.0, 0.971130, 0.0, 0.238550]
```

3. 先生成50条debug轨迹检查碰撞、相机和J5余量，再生成约1000～5000条pick轨迹。
4. 使用`translate_dataset_seq.py`重放转换。
5. 使用`create_lerobot_dataset.py --rotation euler`生成LeRobot v2.1数据集。
6. 从`dynamic-vla-DOM`加载权重，以约`1e-5`学习率、冻结vision/text/connector、20～50 epoch进行第一阶段微调。
7. 仿真评测通过后再部署shadow。
8. 若要解决真实相机域差异，需要补充真机示范采集器，用Gemini 305、D435i、真实model TCP和action标签采集第二阶段数据。公开仓库当前没有本项目可直接使用的真机采集器。

训练命令形态：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py \
  -e piper-new-start-ft \
  -c configs/dynamicvla_piper_finetune.yaml \
  -p /home/yuling/work_space/repos/DynamicVLA/dynamic-vla-DOM \
  -d local/piper-new-start
```

必须保持：

```text
rotation=euler
use_delta_action=true
n_obs_steps=2
history=[-2,0]
chunk_size=20
wrist_cam/opst_cam语义不变
state shape=6
action shape=7
```

### 25.15 当前日志事件全集

除旧文档事件外，当前新增：

| event | 含义 |
|---|---|
| `execute_time_limit` | 达到execute运行时限，停止模型动作更新 |
| `robot_status_fault` | Piper离开CAN/MOVE P/normal，包含逐轴标志和关节角 |
| `return_to_training_start_begin` | 正常结束开始MOVE J回位 |
| `return_to_training_start_complete` | 回位到达稳定，含实际关节和夹爪0 mm目标 |

`action`事件现在包含：

```text
raw
safe
state
executed
robot_status {ctrl_mode, mode_feed, arm_status, motion_status}
```

`runtime_stop.scheduler`包含：

```text
accepted_chunks
bootstrap_chunks
stale_chunks
expired_actions
executed_actions
```

### 25.16 下一位开发者优先级

P0：

1. 不要继续靠固件0x04试错；加入基于URDF/Piper DH的在线IK或目标可达性检查。
2. 对候选IK解检查六轴软限位余量，并保持构型分支连续。
3. 增加目标缩放/跳过策略，但不能在未知碰撞情况下自动反复试探。
4. 明确解决`ctrl_mode=standby`的控制器恢复流程，不得无支撑调用可能失能的reset。
5. 为夹爪增加去抖、最小保持时间和实际开度确认。

P1：

1. 在新起点完成仿真数据生成与微调。
2. 比较真机双视角与仿真训练图像。
3. 增加真实示范采集和sim/real混合微调。
4. 对首次chunk和后续LAAS分别做日志回归。

P2：

1. 碰撞模型、桌面平面、腕部相机和线缆包络。
2. 跟踪误差、速度、加速度和卡滞联锁。
3. 研究MOVE P、MOVE L、MOVE CPV中哪一种更适合25 Hz流式动作。

### 25.17 当前必须保持的新增不变量

1. 第一accepted chunk必须从action[0] bootstrap，不能按首轮推理耗时直接跳到horizon末端。
2. 正常LAAS只能在已经存在旧chunk填充推理空窗后启用。
3. MOVE P前必须预装当前EndPose，避免激活持久化旧目标。
4. 只有真实反馈为CAN + MOVE P + normal时才能开始策略动作。
5. 0x04等状态异常必须在下一控制tick停止，不得继续把`executed=true`当成控制器接受证明。
6. `executed=true`只表示Python调用了SDK，真实执行必须由状态和末端/关节反馈确认。
7. 异常退出不得自动回远处起始点；只保持当前关节。
8. 正常回位和独立回位闭合夹爪，异常退出保持最后夹爪命令。
9. `TRAINING_START_DEG`只允许在`deploy/targets.py`定义一次。
10. 新起点必须同步到仿真、数据和checkpoint后，才能声称模型已适配，而不是只改真机回位常量。
## 26. 主机IK执行后端（2026-06-24，覆盖旧MOVE P执行说明）

### 26.1 当前结论

此前`EndPoseCtrl/MOVE P`在Piper固件内部进行不可见IK，出现`0x04`时只能知道
目标关节超限，不能读取或选择固件候选解。当前默认执行后端已改为
`host_ik_move_j`，旧章节关于MOVE P的内容仅保留为问题历史。

### 26.2 当前代码

- `deploy/kinematics/piper_ik.py`：受限多初值IK、候选过滤与选择；
- `deploy/devices/piper_robot.py`：MOVE J握手、IK预解算、JointCtrl下发；
- `deploy/run.py`：shadow/execute IK日志、拒绝即停和MOVE J状态联锁；
- `deploy/config.py`：IK后端、误差、限位裕量和关节步长参数；
- 三份Piper YAML均显式设置`control_backend: host_ik_move_j`；
- `deploy/tests/test_piper_host_ik.py`：近邻精确解、关节向量整体限幅和不可达目标拒绝测试。

### 26.3 求解与安全语义

IK使用Piper SDK官方`C_PiperForwardKinematics(dh_is_offset)`和SciPy
`least_squares`。边界固定为官方六轴范围，并额外扣除配置的软限位裕量。精确IK候选必须同时满足：

1. link6位置误差不超过`ik_position_tolerance_m`；
2. SO(3)旋转误差不超过`ik_rotation_tolerance_rad`；
3. 所有关节距离硬限位不少于`ik_min_joint_limit_margin_deg`。

若合法精确解相对当前反馈的最大关节变化超过`ik_max_joint_step_deg`，运行时不再拒绝，而是对完整六轴增量使用同一比例缩放，使最大关节变化恰好等于配置上限。不得逐轴独立裁剪。限幅后的关节目标通过官方FK重新计算并记录TCP位置/姿态误差，下一周期从最新反馈继续求解。

`piper_sequential.yaml`启用`ik_allow_pose_projection`。当精确解不存在，或精确解虽然存在但最小关节限位裕量小于`ik_projection_joint_limit_margin_deg`时，主机IK会在更保守的关节边界内重新优化最近可达位姿；位置误差必须不超过`ik_projection_max_position_error_m`，姿态误差必须不超过`ik_projection_max_rotation_error_rad`。这用于处理模型末端pose不可信、J5等关节贴边的情况，例如把接近70度的腕部姿态投影到68度附近，同时记录`pose_projected`和`pose_projection_reason`。

主seed为上次选中解和当前反馈；主seed无合法解才搜索预设多构型seed。选择评分优先
关节连续性，同时惩罚小限位裕量和位姿误差。无候选时抛出`HostIKError`；execute记录
`host_ik_reject`后停止，不发送JointCtrl。shadow只记录，不运动。

### 26.4 控制模式变化

当前正常execute状态必须为：

```text
ctrl_mode == 0x01  # CAN
mode_feed == 0x01  # MOVE J
arm_status == 0x00
```

启动握手通过当前实测六轴执行MOVE J位置保持，不再切入MOVE P，也不会激活旧
EndPose寄存器。每次成功IK发送`MotionCtrl_2(0x01,0x01,speed,0x00)`与
`JointCtrl(J1...J6)`；夹爪仍由独立`GripperCtrl`控制。

### 26.5 当前限制

- 这是自研SciPy受限数值IK，不是官方MoveIt/KDL；
- 使用官方SDK FK，但必须保证`dh_is_offset`与固件/URDF版本一致；
- 没有桌面、相机、线缆、自碰撞模型；
- 多seed不是数学意义上的全部解析逆解枚举；
- 5度关节步长是命令连续性联锁，不等价于路径碰撞安全；
- 未经shadow日志检查不得直接放宽关节步长、误差阈值或限位裕量。

### 26.6 验证状态

离线Piper官方FK闭环测试中，目标`[0,50,-50,0,45,0]`度可由附近反馈恢复，
误差接近数值零；首帧约5 ms，连续warm-start约1.2 ms。新增IK测试4项通过。
起始点测试已同步到当前目标，完整`deploy/tests`共26项全部通过。


### 录像记录

`deploy.run` 支持按 run 自动录像。`runtime.record_video: true` 时，运行目录下会生成：

```text
videos/opst_cam.mp4
videos/wrist_cam.mp4
```

`piper_sequential.yaml` 已设置 `record_video: true`、`video_fps: 25.0`。实现位置：

```text
deploy/common/video_recorder.py
deploy/run.py
```

录像通过 `AsyncVideoWriter` 后台线程写 MP4；相机线程只入队 RGB 帧，队列满时丢录像帧，不能阻塞控制循环。事件日志包含 `video_recording_start`、`video_recording_stop` 和可选 `video_recording_error`。

## 27. 三种IK/执行方案总览与切换设计（权威章节，2026-06-24）

> 本节覆盖第25节中“尚未实现在线IK”的旧结论，并扩充第26节。后续开发者讨论
> Piper末端控制、0x04、主机IK、MoveIt时，应以本节和实际代码为准。必须区分
> “已经实现”“仅保留回退”“尚待实现”，不得把设计方案写成已验证功能。

### 27.1 三条链路的共同输入和坐标契约

DynamicVLA输出固定为：

```text
action = [x, y, z, rx, ry, rz, gripper]
x/y/z：model_tcp在模型基座坐标中的绝对位置，单位m
rx/ry/rz：model_tcp欧拉角，XYZ约定，单位rad
gripper：连续量，部署安全层映射到实际夹爪行程
```

模型目标不是Piper SDK直接使用的link6目标。所有后端都必须经过同一SE(3)转换：

```text
T_base_model_tcp = DynamicVLA action
T_link6_model_tcp = 配置中的固定TCP外参
T_base_link6 = T_base_model_tcp × inverse(T_link6_model_tcp)
```

配置：

```yaml
sdk_to_model_translation_m: [0.0, 0.0, 0.1334]
sdk_to_model_euler_xyz_rad: [0.0, 0.0, 3.141592653589793]
```

禁止任何IK后端直接把model_tcp当成SDK link6，否则位置和姿态均会错误。相机坐标不
直接进入IK；模型已经基于图像输出base/model_tcp动作，IK只消费变换后的base/link6目标。

### 27.2 方案对比

| 项目 | Piper固件IK | 当前SciPy主机IK | MoveIt/KDL |
|---|---|---|---|
| 配置名称 | `firmware_move_p` | `host_ik_move_j` | 计划：`moveit_kdl` |
| 当前状态 | 代码保留，可回退 | 已实现，当前默认 | 尚未接入deploy |
| 求解位置 | Piper控制器固件 | DynamicVLA宿主Python | ROS 2/MoveIt进程 |
| 输入 | SDK link6 XYZ/RPY | SDK link6 XYZ/RPY | Pose + 当前关节/RobotState |
| IK算法 | 官方未公开 | SciPy bounded least-squares | KDL Jacobian/SVD迭代 |
| FK/模型 | 固件内部，不可见 | Piper SDK官方FK | Piper官方URDF + KDL Chain |
| 关节限位 | 固件内部检查，不暴露内部IK解 | 主机显式有界，可做最近可达投影 | URDF bounds + KDL裁剪 |
| 多初值 | 未公开 | 当前/上次解 + 备用seed | 初始seed；超时内随机reseed |
| 候选解可见 | 否 | 是 | 是，可通过服务返回 |
| 选择策略 | 不可控制 | 连续性、误差、限位裕量 | KDL成功解 + MoveIt有效性回调 |
| 碰撞检查 | 未公开，不应假定存在 | 无 | PlanningScene可提供 |
| 路径规划 | 固件内部，不透明 | 无，只发送连续关节目标 | OMPL/规划管线可提供 |
| 下发接口 | MOVE P + EndPoseCtrl | MOVE J + JointCtrl | 关节解/轨迹 → JointCtrl |
| 典型失败 | 0x02/0x03/0x04 | HostIKError | NO_IK_SOLUTION/PLANNING_FAILED |
| 适合用途 | 官方基线、小范围诊断 | 无ROS低延迟、shadow、备用 | 正式碰撞感知规划 |

### 27.3 方案A：Piper固件IK（firmware_move_p）

#### 27.3.1 控制路径

```python
MotionCtrl_2(0x01, 0x00, speed_percent, 0x00)  # MOVE P
EndPoseCtrl(X, Y, Z, RX, RY, RZ)
GripperCtrl(...)  # 独立
```

CAN目标：

```text
0x152：X/Y
0x153：Z/RX
0x154：RY/RZ
```

固件接收完整link6位姿，在机械臂主控内部完成IK、限位判断和运动处理。

#### 27.3.2 官方可观测信息

`GetArmStatus()`只能返回：

```text
0x02：无解
0x03：奇异点
0x04：目标角度超过限位
err_status.joint_1_angle_limit ... joint_6_angle_limit
```

`GetArmJointMsgs()`是实际反馈关节角，不是固件IK目标；`GetArmJointCtrl()`读取
CAN 0x155～0x157关节控制命令，也不是MOVE P内部IK结果；`GetFK("control")`
对上次JointCtrl命令做正解，仍不能读取EndPoseCtrl内部逆解。

不存在官方公开的：

```text
GetIKTargetJoints
指定elbow-up/down
指定wrist-flip
传入IK seed
枚举全部逆解
失败后修改单轴目标
```

因此固件返回0x04后，应用层没有“把J5裁剪到69度再继续”的操作窗口。即使可以裁剪，
单独修改J5也会破坏目标末端位姿，正确做法应是寻找另一组完整六轴解。

#### 27.3.3 已知握手问题

EndPose寄存器跨进程保留旧值。直接从MOVE J切入MOVE P可能激活旧目标并立刻产生
0x04。旧固件后端因此必须先写入当前实测EndPose覆盖寄存器，再重复发送模式和当前目标，
直到反馈为：

```text
ctrl_mode == 0x01
mode_feed == 0x00
arm_status == 0x00
```

该逻辑仍保留在`PiperRobot.enable_motion()`的`firmware_move_p`分支，只用于
回退和对照；当前默认配置不会进入该分支。

#### 27.3.4 优缺点和适用范围

优点：

- 官方控制器原生支持；
- 应用代码简单；
- 控制器内部负责底层执行。

缺点：

- IK算法、seed、候选解、分支选择全部黑盒；
- 无法证明0x04前已经搜索全部合法构型；
- 无法记录“固件算出的J5具体是多少”；
- 无法在发送前执行主机侧关节连续性和软限位筛选；
- 不应假定固件包含桌面、相机、线缆或完整自碰撞模型。

只建议用于官方基线对照、1mm笛卡尔诊断或主机IK故障时的受控回退，不再作为本项目
默认执行后端。

### 27.4 方案B：当前自研SciPy主机IK（host_ik_move_j）

#### 27.4.1 实现位置和依赖

```text
deploy/kinematics/piper_ik.py
deploy/devices/piper_robot.py
deploy/run.py
deploy/config.py
deploy/tests/test_piper_host_ik.py
```

依赖：

- `piper_sdk.C_PiperForwardKinematics`：官方Piper正运动学；
- `scipy.optimize.least_squares`：有界非线性最小二乘；
- `scipy.spatial.transform.Rotation`：SO(3)旋转误差。

它不是Piper官方IK，也不是MoveIt/KDL。准确表述必须是：

```text
Piper官方FK/关节参数 + 本项目自研受限数值IK + Piper官方JointCtrl
```

#### 27.4.2 数学目标

对于关节向量q：

```text
position_error = FK_position(q) - target_position
rotation_error = Log(R_target × inverse(R_FK(q)))
residual = [
  position_error / position_tolerance,
  rotation_error / rotation_tolerance
]
```

求解器在软化后的官方关节边界内最小化该残差。官方硬限位：

```text
J1 [-150, 150] deg
J2 [0, 180] deg
J3 [-170, 0] deg
J4 [-100, 100] deg
J5 [-70, 70] deg
J6 [-120, 120] deg
```

实际优化边界还会扣除`ik_min_joint_limit_margin_deg`。

#### 27.4.3 seed和分支策略

优先seed：

1. 上一个已选主机IK解；
2. 当前实际关节反馈。

主seed满足全部安全条件时立即使用，以维持关节连续性和低延迟。主seed失败后尝试若干
预设肩/肘/腕构型seed。当前实现是多初值数值搜索，不是解析法全部解枚举；仍可能漏掉
某些远离seed的合法分支。

候选评分包含：

- 相对当前关节的归一化距离；
- 距离关节硬限位的最小裕量；
- 位置误差；
- SO(3)旋转误差。

#### 27.4.4 下发前硬条件

精确IK候选必须满足：

```text
position_error <= ik_position_tolerance_m
rotation_error <= ik_rotation_tolerance_rad
minimum_limit_margin >= ik_min_joint_limit_margin_deg
```

发送前按以下方式限制关节命令：

```text
delta = q_ik - q_feedback
scale = min(1, ik_max_joint_step_deg / max(abs(delta)))
q_command = q_feedback + scale * delta
```

当前配置：

```yaml
control_backend: host_ik_move_j
ik_position_tolerance_m: 0.002
ik_rotation_tolerance_rad: 0.035
ik_max_joint_step_deg: 5.0
ik_min_joint_limit_margin_deg: 0.2
ik_max_nfev: 60
```

5度是单周期关节命令的整体向量限幅，不是速度。限幅后目标不再精确实现本周期TCP，因此日志同时记录精确IK目标、实际发送关节目标和限幅后FK误差。机械臂实际速度仍由
`command_speed_percent`限制。禁止因为“机械臂看起来没动”就同时放宽步长、速度和
误差阈值。

#### 27.4.5 执行路径

```python
MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)  # MOVE J
JointCtrl(j1, j2, j3, j4, j5, j6)
GripperCtrl(...)  # 若配置启用
```

正常状态联锁：

```text
ctrl_mode == 0x01
mode_feed == 0x01
arm_status == 0x00
```

启动时通过实测当前六轴执行MOVE J保持，不进入MOVE P，因此不会激活持久化的旧
EndPose目标。

#### 27.4.6 shadow、日志和失败语义

shadow会运行同一IK但不使能、不发送JointCtrl。关键字段：

```text
action.planned_command.model_tcp
action.planned_command.sdk_link6
action.robot_current.joint_degrees
action.host_ik.candidates[]
action.host_ik.selected_joint_degrees
action.host_ik.selected.position_error_m
action.host_ik.selected.rotation_error_rad
action.host_ik.selected.minimum_limit_margin_deg
action.host_ik.selected.maximum_step_from_current_deg
host_ik_solution
host_ik_reject
previous_command_feedback.ik_realized_joint_degrees
```

无安全候选时抛出`HostIKError`。shadow记录拒绝并继续观察；execute记录拒绝后立即
停止，当前命令不会发送。

#### 27.4.7 当前验证和限制

已验证：

- 官方FK闭环目标可从附近关节恢复；
- 首次/近邻解约5ms，warm-start约1.2ms；
- 完整deploy测试26项通过。

未验证或未提供：

- 没有解析式全部分支枚举；
- 没有桌面、相机、线缆和完整自碰撞；
- 没有OMPL路径搜索；
- 没有速度/加速度轨迹时间参数化；
- 尚未完成该新后端的真机运动验收。

因此当前SciPy后端是可观察、低延迟的诊断/备用实现，不应宣称等价于MoveIt。

### 27.5 方案C：MoveIt 2 + KDL（moveit_kdl，待实现）

#### 27.5.1 ROS、MoveIt、KDL各自职责

```text
ROS 2：进程、消息、服务、TF和生命周期通信
MoveIt 2：RobotModel、PlanningScene、规划请求、碰撞检查、轨迹和执行框架
KDLKinematicsPlugin：MoveIt调用的数值FK/IK插件
OMPL：可选的关节空间路径规划器
Piper ROS节点或本项目适配器：把关节目标/轨迹转换为JointCtrl
```

ROS本身不计算IK；真正IK由KDL插件完成。只复用KDL可以不使用ROS，但将失去MoveIt的
PlanningScene、碰撞检查和完整规划管线。

#### 27.5.2 Piper官方配置

Piper Humble官方MoveIt配置使用：

```yaml
kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
kinematics_solver_search_resolution: 0.005
kinematics_solver_timeout: 0.005
```

KDL插件从Piper URDF创建base到tip的KDL Chain，读取URDF关节上下限。典型求解：

1. 使用当前关节作为初始seed；
2. FK计算当前末端；
3. 计算目标与当前之间的Cartesian Twist误差；
4. 使用Jacobian SVD得到关节增量；
5. 将增量裁剪到关节上下限；
6. 误差增大/接近奇异点时减小步长；
7. 卡住时执行小扰动；
8. 首次失败后在超时允许范围内随机reseed；
9. 将解交给MoveIt有效性回调继续做约束/碰撞检查。

KDL同样是seed相关数值IK，不保证数学意义上枚举所有解析分支；它的优势是成熟、使用
官方URDF、可接MoveIt完整有效性检查，而不是“必然找到所有解”。

#### 27.5.3 两种MoveIt使用模式

IK-only：

```text
TCP + 当前六轴
→ MoveIt/KDL求一个受限关节解
→ 返回本项目
→ 本项目做步长/状态联锁
→ JointCtrl
```

适合25Hz连续小动作，但仍需要latest-only和超时机制。

Full planning：

```text
TCP目标
→ KDL采样合法goal state
→ PlanningScene碰撞过滤
→ OMPL搜索当前q到目标q路径
→ 时间参数化
→ 返回完整JointTrajectory
→ 缓冲执行
```

适合起始点回位、大范围绕障和需要桌面/自碰撞检查的动作，不适合每40ms重新做一次
完整规划。

#### 27.5.4 建议的后端接口

未来不要在`run.py`中直接写ROS调用。应新增抽象层：

```python
class MotionBackend:
    def prepare(target_tcp, robot_state) -> PreparedMotion
    def execute(prepared_motion, robot) -> CommandReceipt
```

三种实现：

```text
FirmwareMovePBackend
ScipyHostIKMoveJBackend
MoveItKDLBackend
```

建议配置：

```yaml
robot:
  control_backend: moveit_kdl

moveit:
  endpoint: http://127.0.0.1:8765
  mode: ik_only              # ik_only | planned_trajectory
  request_timeout_ms: 30
  max_result_age_ms: 100
  cancel_superseded: true
```

MoveIt进程只返回关节解/轨迹，不应直接拥有CAN。CAN继续由`deploy.run`唯一控制，避免
Piper ROS节点与Python适配器同时下发命令。

#### 27.5.5 开源和复用边界

- MoveIt 2主体为BSD 3-Clause；
- MoveIt KDL插件源码为BSD；
- Orocos KDL为LGPL-2.1-or-later；
- 可以直接链接、封装或在遵守许可证的前提下修改；
- 对IK-only，优先直接使用独立KDL/PyKDL或小型C++/pybind服务，不建议照抄算法；
- 对碰撞和规划，不建议重写MoveIt核心：RobotModel、FCL、PlanningScene、OMPL和轨迹
  参数化的组合远大于一个IK函数。

许可证不是法律意见；若项目需要闭源分发，必须单独审查LGPL动态链接和再分发义务。

### 27.6 Ubuntu 24.04安装决策

宿主机实测：

```text
Ubuntu 24.04.4 LTS noble
x86_64
Docker当前未安装
```

Piper官方仓库分支：

```text
noetic：Ubuntu 20.04
foxy
humble：Ubuntu 22.04，官方MoveIt 2路径
没有Jazzy/Ubuntu 24.04分支
```

不要在Ubuntu 24.04宿主机强行安装Noetic或未经官方验证地直接编译Humble。建议：

```text
Ubuntu 24.04宿主：DynamicVLA、相机、CAN、deploy.run
Docker Ubuntu 22.04：ROS 2 Humble、Piper URDF、MoveIt 2、KDL、规划服务
本机网络接口：HTTP/gRPC/Unix socket，latest-only请求
CAN所有权：只属于宿主deploy.run
```

完整安装命令见当前对话记录；核心容器镜像为
`osrf/ros:humble-desktop-full`，Piper仓库使用`humble`分支。安装后先验证包和
规划服务，不给容器CAN权限，不启动自动使能真机节点。

### 27.7 延迟预算与异步设计

已测模型推理均值约266ms；控制循环25Hz，即40ms/tick。需要区分：

```text
ROS/DDS本机通信延迟
KDL IK计算时间
PlanningScene碰撞检查时间
OMPL全路径规划时间
模型推理时间
```

本机ROS通信通常不是最大项，但必须实测，不能写死假设。官方Piper KDL超时配置为5ms；
完整MoveIt规划可能从几十毫秒到数秒，绝不能在25Hz循环中阻塞调用。

MoveIt后端必须异步：

```text
控制线程提交(index, timestamp, TCP, current_q)
→ MoveIt worker只保留最新请求
→ 新请求到达时取消/丢弃旧请求
→ 返回结果携带source_index和完成时间
→ 运行时按max_result_age_ms拒绝过期解
→ 执行前再次比较当前q、关节步长和机器人状态
```

对于连续小幅VLA动作，优先IK-only；对于回位或大范围运动，暂停策略流，使用完整规划
并执行固定轨迹。不得把LAAS跳步直接套到长时间MoveIt轨迹内部。

### 27.8 三后端验收矩阵

每个后端必须分别通过，不能用一个后端的结果替代另一个：

1. 纯离线：已知q → FK目标 → IK回算 → 误差和限位；
2. 同一TCP、不同初始构型：记录是否选择不同分支；
3. 已知不可达目标：必须拒绝且不发送CAN；
4. J5边界目标：记录候选解、限位裕量和拒绝原因；
5. shadow 5分钟：无求解线程崩溃、无过期结果被接受；
6. 单步真机：1%速度、固定小目标、现场急停；
7. 连续真机：检查目标关节、反馈关节和TCP跟踪；
8. MoveIt full planning：加入桌面碰撞体，验证规划不会穿桌；
9. 后端切换：进程重启后模式握手必须与后端一致；
10. CAN独占：任何时刻只能有一个进程调用Piper控制接口。

### 27.9 当前开发状态和下一步

当前真实状态：

```text
firmware_move_p：已实现并保留，非默认
host_ik_move_j：已实现、测试26项通过、当前默认，尚待新后端真机验收
moveit_kdl：只完成官方资料、安装路径和架构设计，代码尚未实现
```

下一步顺序：

1. 安装Docker、ROS 2 Humble、MoveIt 2和Piper humble分支；
2. 在容器内做planning-only验证，不接CAN；
3. 实现只读IK服务：输入link6 Pose + current_q，输出候选q和误差；
4. 与当前SciPy IK对同一批日志目标做离线对比；
5. 增加`moveit_kdl`后端和latest-only异步客户端；
6. shadow检查延迟、解分支、J5余量；
7. 最后才进行低速真机JointCtrl测试；
8. 碰撞模型完成后再启用full planning。

