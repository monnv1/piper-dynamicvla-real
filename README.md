# DynamicVLA 真机部署

本目录为 Gemini 305、RealSense D435i 和 AgileX Piper 提供解耦的真机运行时。默认模式为 `shadow`：读取真实设备并执行模型推理，但绝不向机械臂发送运动、末端位姿或夹爪命令。

> 当前代码已通过无硬件单元测试，但尚未完成真实模型、相机和机械臂联合执行验证。首次使用必须从只读诊断和 shadow 模式开始。配置中的工作空间是保守占位值，不能直接作为实际设备安全边界。

## 已实现功能

- Gemini 305 RGB采集，使用 `pyorbbecsdk2`。
- D435i RGB采集，使用 `pyrealsense2`。
- Piper关节、末端位姿和夹爪反馈读取。
- 双相机基于主机单调时钟的软件同步。
- 25 Hz观测组装，默认使用论文中的 `{t-2, t}` 两帧历史。
- DynamicVLA异步推理；等待推理时控制循环不会阻塞。
- 基于episode、观测时间戳和控制步的LAAS动作过期处理。
- 工作空间、有限值、单步平移、单步旋转和夹爪范围过滤。
- shadow与execute隔离，execute需要配置和命令行双重确认。
- 每次运行生成JSONL事件日志，并提供延迟/LAAS汇总工具。

## 目录和模块边界

```text
deploy/
├── common/       # 数据消息、最新值、帧缓冲、事件日志
├── configs/      # 外部可修改的YAML配置
├── control/      # 与模型和SDK无关的安全过滤
├── devices/      # Orbbec、RealSense、Piper厂商适配层
├── kinematics/   # 主机端IK求解器（SciPy least-squares、Differential、Pink/Pinocchio）
├── policy/       # 观测组装、异步模型和LAAS
├── tests/        # 不需要硬件的单元测试
├── tools/        # 工具集：日志分析、诊断、单关节/位姿移动、遥操作、离线评测等
├── config.py     # 类型安全的YAML配置加载与校验（~100项检查）
├── diagnose.py   # 只读硬件诊断
├── run.py        # 真机运行入口
└── targets.py    # 训练开始关节角常量
```

设备模块只负责标准化数据；模型不直接导入厂商SDK；Piper适配层不依赖DynamicVLA。替换相机、机器人或调度器时，不需要修改其他层。

## 外部接口

主要配置文件是 `deploy/configs/piper_gemini_d435i.yaml`。

### 预置配置文件

| 文件名 | 模式 | 后端 | 用途 |
|---|---|---|---|
| `piper_gemini_d435i.yaml` | shadow | host_ik_move_j | 标准影子模式（默认） |
| `piper_gemini_d435i_first_execute.yaml` | execute | host_ik_move_j | 首次执行（30s上限，保守限幅） |
| `piper_sequential.yaml` | execute | firmware_move_p | 40 Hz 定时下发，无 LAAS 重叠 |
| `piper_model_tcp_axis_diagnostic.yaml` | shadow | firmware_move_p | 固件 MOVE P 笛卡尔轴诊断 |
| `piper_model_tcp_axis_hostik_diagnostic.yaml` | shadow | host_ik_move_j | 主机 IK 笛卡尔轴诊断 |

### 预训练模型路径

可写入YAML：

```yaml
model:
  checkpoint: /absolute/path/to/dynamic-vla-DOM
```

也可以在启动时覆盖：

```bash
python -m deploy.run \
  --checkpoint /absolute/path/to/dynamic-vla-DOM \
  --mode shadow
```

模型目录至少应包含：

```text
dynamic-vla-DOM/
├── config.json
└── model.safetensors
```

运行时从 `config.json`读取输入特征、动作维度、归一化统计、chunk长度和delta action设置。当前Piper适配要求模型使用Euler动作表示：

```text
[x, y, z, rx, ry, rz, gripper]
```

位置单位为米，Euler角单位为弧度。运行时以checkpoint的 `input_features`为准自动适配状态维度：当前DOM checkpoint声明6维state，因此输入为 `[x,y,z,rx,ry,rz]`；7维action为 `[x,y,z,rx,ry,rz,gripper]`，其中模型夹爪值为 `+1=打开、-1=闭合`，安全层再将其映射为Piper的米制开口量。

### Piper坐标系与TCP转换

策略状态、策略动作和安全工作空间统一使用 `base_link -> model_tcp`。Piper SDK反馈和 `EndPoseCtrl`使用 `base_link -> link6`。适配器在设备边界进行双向SE(3)转换，策略层和安全层不会接触SDK坐标：

```text
T_base_model_tcp = T_base_sdk * T_sdk_model_tcp
T_base_sdk       = T_base_model_tcp * inverse(T_sdk_model_tcp)
```

原始仿真URDF定义 `link6 -> gripper_base = Rz(pi)`，训练数据追踪点又位于 `gripper_base`局部 `+Z 0.1334 m`，因此YAML默认值为：

```yaml
sdk_to_model_translation_m: [0.0, 0.0, 0.1334]
sdk_to_model_euler_xyz_rad: [0.0, 0.0, 3.141592653589793]
```

反馈先转换为模型TCP再进入归一化和安全检查；模型动作通过安全检查后再逆变换为SDK link6目标。模型Euler的X、Z反馈会按原始数据预处理包装到 `[0, 2*pi)`。如果现场验证发现固件EndPose已经包含工具TCP，则必须用实测变换替换上述值，不能同时在固件和本适配器重复补偿。

### 相机接口

```yaml
cameras:
  opst_cam:
    driver: orbbec
    serial: "Gemini序列号；只有一台时可留空"
  wrist_cam:
    driver: realsense
    serial: "D435i序列号；只有一台时可留空"
```

模型输入键固定映射为：

```text
D435i      -> observation.images.opst_cam  (固定第三视角)
Gemini 305 -> observation.images.wrist_cam (腕部视角)
```

两个适配器均输出RGB顺序的 `uint8[H,W,3]`，不是OpenCV BGR。

### Piper接口

```yaml
robot:
  can_interface: can0
  dh_is_offset: 1
  auto_enable: false
  command_gripper: false
  command_speed_percent: 10
```

`auto_enable: false`时运行时不能进入execute。启动Piper只调用CAN连接和反馈读取；只有execute双重确认通过后才调用 `EnableArm`、`MotionCtrl_2`和 `EndPoseCtrl`。夹爪还有独立的 `command_gripper`开关，默认关闭；开启后才调用 `GripperCtrl`。

`feedback_pose_source` 控制 `RobotState.model_vector()` 的末端位姿来源：
- `endpose`（默认）：直接使用固件 EndPose 反馈。
- `fk`：对关节反馈执行主机正解（Piper SDK `C_PiperForwardKinematics`），不依赖固件 EndPose。

### 执行后端

`control_backend` 选择运动执行链路：

| 后端 | 说明 |
|---|---|
| `host_ik_move_j` | SciPy least-squares 多初值主机 IK → MOVE J |
| `host_diff_ik_move_j` | 阻尼最小二乘微分 IK（IsaacLab 风格）→ MOVE J |
| `host_pink_ik_move_j` | Pink + Pinocchio 微分 IK，直接在模型 TCP 帧求解 → MOVE J |
| `firmware_move_p` | 直接下发笛卡尔位姿目标，由固件黑盒 IK 执行 MOVE P |

各个配置 YAML 可以选择不同的后端。shadow 模式对所有后端都执行 IK 预览和日志诊断。

## 安装

在项目Python 3.10环境内：

```bash
conda activate dynamicvla
python -m pip install -r requirements.txt
python -m pip install -r deploy/requirements.txt
```

udev规则只需在系统中安装一次。CAN接口必须为1 Mbps：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

## 启动顺序

### 1. 设备和只读反馈

连接Piper USB-CAN、Gemini 305和D435i，确认CAN为 `UP/ERROR-ACTIVE`，然后运行：

```bash
python -m deploy.diagnose --seconds 10
```

该命令不会加载模型、使能机械臂或发送动作。输出应持续包含双相机帧号、图像形状、Piper反馈频率、关节角和末端位置。

### 2. 配置shadow

修改：

- `model.checkpoint`
- 两台相机的serial
- `model.task`
- `dh_is_offset`
- 实际测量的工作空间边界
- `gripper_open_threshold`以及Piper实际开合行程

保持：

```yaml
runtime:
  mode: shadow
robot:
  auto_enable: false
```

### 3. 启动shadow推理

```bash
python -m deploy.run \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --mode shadow
```

按 `Ctrl+C`停止。日志写入：

```text
deploy/runs/<日期时间-episode>/events.jsonl
```

### 4. 分析日志

```bash
python -m deploy.tools.summarize_log \
  deploy/runs/<episode>/events.jsonl
```

重点检查：

- `safety_reject`必须为0或原因明确。
- 推理延迟应小于20个控制周期，即0.8秒。
- `laas_skipped_steps`应与推理延迟对应，例如0.226秒约跳过5至6步。
- 模型预测位置应位于Piper实际工作空间。
- raw action与当前末端位姿的坐标轴方向必须一致。

### 5. execute前置条件

只有完成 `deploy/DEBUGGING.md` 中的检查表后才能：

```yaml
runtime:
  mode: execute
robot:
  auto_enable: true
```

execute仍需要命令行确认：

```bash
python -m deploy.run \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --mode execute \
  --confirm-motion
```

缺少任意一个开关都会拒绝启动。

### Execute 运行时选项

```yaml
runtime:
  max_execute_seconds: 60        # 执行时长上限（秒），0 表示不限
  return_to_training_start_on_normal_exit: true  # 正常退出后自动回到训练开始位姿
  return_speed_percent: 3        # 返回运动速度 1-10%
  return_timeout_s: 45           # 返回运动超时
```

设置 `max_execute_seconds` 可以让 execute 在指定时间后自动停止并返回起始位。异常退出（异常、Ctrl+C）不会触发自动返回。

Execute 模式下运行时还持续监控 Piper CAN 状态：

| 监控项 | 条件 | 行为 |
|---|---|---|
| `ctrl_mode` | 非 `0x01` (MOTION_ENABLE) | 立即报错退出 |
| `mode_feed` | 非期望值 | 立即报错退出 |
| `arm_status` | 非 `0x00` (STANDBY) | 立即报错退出 |
| `joint_limit_flags` | 任一关节触发硬限位 | 日志记录后退出 |
| `motion_status` | 非 `0x00` 时表示运动中 | 用于完成检测 |

### 顺序执行模式（Sequential）

默认 `continuous_inference: true` 使用 LAAS 重叠推理和执行。设置 `continuous_inference: false` 后，每个 chunk 依次执行，并通过 `action_execution_mode` 选择两种下发方式：

- `timed`：按 `action_hz` 固定时间间隔发送 setpoint，不等待从臂到位；控制循环超时时不会突发补发漏掉的点。
- `point_to_point`：等待固件停止且关节误差稳定后，再发送下一个 setpoint，保留原来的执行行为。

```yaml
runtime:
  continuous_inference: false
  max_trusted_action_steps: 20      # 每次推理使用的前 N 步
  action_execution_mode: timed      # timed 或 point_to_point
  control_hz: 40.0                  # 必须不低于 action_hz
  action_hz: 40.0                   # 每 25 ms 下发一个动作
  action_completion_joint_tolerance_deg: 0.5  # 关节到达判定容差
  action_completion_settle_cycles: 3          # 稳定判定所需连续周期数
  action_completion_timeout_s: 30.0           # 单步超时
```

三个 `action_completion_*` 参数只在 `point_to_point` 模式生效。命令行可临时覆盖：

```bash
python -m deploy.run \
  --config deploy/configs/piper_sequential.yaml \
  --action-execution-mode point_to_point \
  --checkpoint /path/to/checkpoint \
  --mode execute \
  --confirm-motion
```

### 动作流保护

```yaml
safety:
  hold_on_stale_action: true   # 模型输出断档时保持最后目标等待，而非报错
  stale_action_hold_ms: 200    # 超过此时间无新动作则触发保护
```

`hold_on_stale_action` 让运行时在模型推理偶发延迟时保持最后 JointCtrl 目标继续等待，而不是直接退出。默认 `false`（动作断档即退出）。

## 工具集

`tools/` 提供以下硬件和离线工具，均在 `deploy/tools/` 下：

| 工具 | 说明 | 硬件需求 |
|---|---|---|
| `summarize_log.py` | 统计 events.jsonl 的事件数、推理延迟、LAAS跳过步数 | 无 |
| `verify_piper_frames.py` | 采样 EndPose vs FK joint6，验证 `sdk_to_model` 变换准确性 | Piper CAN |
| `diagnose_piper_cartesian.py` | MOVE P 笛卡尔诊断：发 ~1 mm 目标并测量反馈一致性 | Piper CAN |
| `enter_teach_mode.py` | 读/切换 Piper 控制模式：STANDBY、MOTION_ENABLE、TEACH | Piper CAN |
| `jog_piper_joint.py` | 单关节低速点动，校验软限位和 FK 路径 | Piper CAN |
| `jog_model_tcp_axis.py` | 沿模型 TCP 坐标轴方向微小点动，用 host IK 执行 | Piper CAN |
| `move_to_joint_zero.py` | 慢速 MOVE J 回到 `[0,0,0,0,0,0]` | Piper CAN |
| `move_to_joint_pose_and_print_tcp.py` | 移动到指定关节角后持续打印模型 TCP 位姿 | Piper CAN |
| `move_to_model_tcp_pose.py` | 移动到指定模型 TCP 位置/四元数，用 host IK | Piper CAN |
| `move_to_training_start.py` | 慢速 MOVE J 回到训练开始关节角，带 FK 路径检查 | Piper CAN |
| `teleop_model_tcp.py` | 终端交互式遥操作模型 TCP 位姿，使用 Pink IK | Piper CAN |
| `print_gripper_state.py` | 只读打印夹爪角度/力/状态 | Piper CAN |
| `counterfactual_rollout.py` | 冻结图像，把上一个模型输出作为下一帧 state，闭环验证 | Piper CAN |
| `replay_real_episode.py` | 回放 LeRobot episode 的绝对模型 TCP 动作，用 host IK | Piper CAN |
| `offline_policy_rollout.py` | 在离线 LeRobot parquet 上运行策略，比较预测 vs 数据集动作 | 无 |
| `offline_policy_dataloader_check.py` | 在 DataLoader sample 上运行策略，验证 delta 动作一致性 | 无 |

所有工具支持 `--help` 查看参数。需要硬件连接的工具在 dry-run 阶段不会发送运动。

## 功能实现原理

### 相机同步

每个相机线程收到帧时记录 `time.monotonic_ns()`。25 Hz观测线程以同一个时钟选择容差内最接近的帧。默认50 ms容差覆盖30 FPS相机约33 ms的最坏到帧间隔。跨厂商设备不假设设备时间戳属于同一时钟域。

### 时序观测

控制循环每40 ms产生一个sample。默认 `history_indices: [-2, 0]`，将当前帧与约80 ms前的帧送入模型，从视觉变化中提供短期速度信息。

### Continuous Inference

推理线程使用单槽输入队列。模型忙时，新观测覆盖尚未处理的旧观测，因此不会形成越来越长的推理积压。控制循环持续25 Hz运行。

### 时间戳LAAS

动作块记录产生它的观测步 `k`。若推理完成时系统在 `c`：

```text
skip = c - k
```

前 `skip` 个动作已过期并删除。若整个20步chunk均过期，则丢弃整块。新chunk只替换尚未执行的未来动作，并拒绝旧episode、乱序chunk和超过最大年龄的chunk。

### 安全过滤

每个动作先检查有限值和目标工作空间。单步平移和旋转过大时按配置限幅，并再次检查限幅后的实际下发点；当前位姿在workspace外时不会借限步逻辑逐步进入，而是拒绝执行，要求先用受控方式回到安全起始位。夹爪的±1标志通过阈值映射到 `gripper_min_m/gripper_max_m`。execute中任何workspace拒绝都会终止运行。

## 与论文和原版代码的区别

| 项目 | 论文/原版仓库 | 本部署实现 |
|---|---|---|
| 环境 | Isaac Lab评测服务 | Gemini、D435i、Piper真机SDK |
| 观测通信 | ZeroMQ pickle | 进程内标准消息和时间戳缓冲 |
| 异步输入 | `Manager().dict()`后清空 | 单槽线程安全队列，覆盖待处理旧观测 |
| LAAS依据 | 离散 `index` | episode + index +主机单调时间戳 |
| 动作合并 | 原实现存在边界和重叠风险 | 新chunk明确拥有其起始点之后的未来动作 |
| 超过chunk延迟 | 可能产生空列表边界问题 | 整块丢弃并记录统计 |
| 安全 | 仿真工作空间终止 | 真机工作空间、步长、角度、有限值和双重使能 |
| 日志 | 仿真结果与pickle | 每周期JSONL，可复盘raw/safe action和延迟 |
| 推理隔离 | multiprocessing子进程 | 独立推理线程；设备层与模型层仍解耦 |
| 真机感知系统 | 论文描述但未公开 | 当前只实现VLA所需双RGB；未实现EfficientTAM物体状态估计 |

论文声称的约88 Hz对应20个动作除以约0.226秒的动作点吞吐率。完整chunk重规划仍约4.4 Hz，实际动作调度按25 Hz运行。

## 当前未完成或必须现场确认

- 用 `GetFK("feedback")[-1]`确认当前固件EndPose等于SDK link6；若固件配置了额外工具TCP，更新YAML固定变换。
- `dh_is_offset`应与机械臂固件版本匹配。
- Gemini 305腕部安装姿态和D435i固定视角是否接近训练数据。
- 真机数据归一化统计是否包含当前Piper embodiment。
- 夹爪反馈和模型夹爪动作的开闭方向。
- 工作空间边界、桌面高度、最大允许速度。
- 厂商控制器对25 Hz绝对笛卡尔目标的平滑性；必要时增加独立插值层。
- 联合只读诊断尚未在本次代码实现后完成：验证时硬件已从USB和CAN总线拔下。
- 论文中的物体6D状态估计和自动数据采集状态机不属于当前VLA执行路径。

在这些项目完成前，只能运行diagnose和shadow。

## 每次运行录像

`deploy.run` 会在 `runtime.record_video: true` 时把每个已启用相机录成 MP4，保存到本次 run 目录：

```text
deploy/runs/<episode_id>/videos/opst_cam.mp4
deploy/runs/<episode_id>/videos/wrist_cam.mp4
```

当前 `piper_sequential.yaml` 已启用：

```yaml
runtime:
  record_video: true
  video_fps: 25.0
```

录像写盘在后台线程完成；如果编码队列满，会丢录像帧但不阻塞控制循环。日志事件：

- `video_recording_start`：本次视频路径；
- `video_recording_stop`：每路相机写入/丢弃帧数；
- `video_recording_error`：编码器异常。

## 主机端 IK + MOVE J（当前默认执行后端）

当前真机执行链路已经从 Piper 固件黑盒 IK 切换为：

```text
DynamicVLA model_tcp目标
→ T_sdk_model_tcp逆变换得到SDK link6目标
→ 主机PiperHostIK受限多初值求解
→ 位置/旋转误差、六轴限位裕量、单周期关节跳变检查
→ MotionCtrl_2(MOVE J) + JointCtrl
→ 独立GripperCtrl
```

配置入口位于`robot`：

```yaml
control_backend: host_ik_move_j
ik_position_tolerance_m: 0.002
ik_rotation_tolerance_rad: 0.035
ik_max_joint_step_deg: 5.0
ik_min_joint_limit_margin_deg: 0.2
ik_max_nfev: 60
# piper_sequential.yaml also enables conservative pose projection:
ik_allow_pose_projection: true
ik_projection_joint_limit_margin_deg: 2.0
ik_projection_max_position_error_m: 0.003
ik_projection_max_rotation_error_rad: 0.08
```

实现使用Piper SDK官方`C_PiperForwardKinematics`作为正解模型，以SciPy有界
least-squares执行主机逆解。连续周期优先使用上次解/当前关节作为seed；主seed失败
后才尝试多组构型seed。它不是Piper固件IK，也不是MoveIt/KDL。

建议先运行shadow；shadow不会使能或发送关节命令，但会生成`host_ik`和
`host_ik_reject`诊断。execute仅在找到满足全部约束的解时发送JointCtrl：

```bash
python -m deploy.run --config deploy/configs/piper_gemini_d435i.yaml --mode shadow

python -m deploy.run \\
  --config deploy/configs/piper_sequential.yaml \\
  --mode execute --confirm-motion
```

关键日志：

- `action.host_ik.candidates`：所有实际尝试的seed、关节解、误差和拒绝原因；
- `action.host_ik.selected_joint_degrees`：准备下发的六轴目标；
- `host_ik_solution`：已经发送给JointCtrl的解；
- `joint_step_limited`：精确IK目标超过单周期关节步长，记录整体缩放比例、实际关节命令和限幅后FK误差；
- `host_ik_reject`：无安全解，execute立即停止且不发送该目标；
- `action_stream_stale_hold`：仅在`hold_on_stale_action: true`时出现，表示模型动作流暂时断档，运行时保持最后JointCtrl目标并等待下一批action；
- `robot_current.joint_degrees`：求解前反馈关节角；
- `previous_command_feedback.ik_realized_joint_degrees`：上一命令后的实际关节反馈。

注意：`ik_max_joint_step_deg`是六轴关节增量整体缩放上限，不是速度参数；超过时不再终止，而是保持关节运动方向并按统一比例缩放。`piper_sequential.yaml`额外启用`ik_allow_pose_projection`：当精确末端姿态会贴近关节限位或无精确安全解时，主机IK会在更保守的关节边界内重优化最近可达位姿，优先保位置、允许小幅牺牲姿态，并记录`pose_projected`。Piper的
`command_speed_percent`仍限制控制器运动速度。`piper_sequential.yaml`还启用`hold_on_stale_action: true`：模型推理偶发超过chunk覆盖时间时不退出，而是保持最后JointCtrl目标等待下一批action。主机IK目前没有环境碰撞模型，不能替代
桌面、相机、线缆和自碰撞检查。

### Pink + Pinocchio 微分 IK（`host_pink_ik_move_j`）

默认值 `control_backend: host_pink_ik_move_j` 使用 [Pink](https://github.com/stephane-caron/pink) + [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 在模型 TCP 帧直接求解微分 IK：

```yaml
robot:
  control_backend: host_pink_ik_move_j
  pink_urdf_path: simulations/robots/PIPER/piper_description.urdf
  pink_frame_name: model_tcp
  pink_solver: proxqp          # proxqp | quadprog
  pink_dt: 0.04                # 积分步长（秒）
  pink_position_cost: 1.0      # 位置误差权重
  pink_orientation_cost: 0.25  # 姿态误差权重
  pink_posture_cost: 0.01      # 关节姿态正则化权重
  pink_lm_damping: 0.0001      # Levenberg-Marquardt 阻尼
  pink_qpsolver_damping: 1e-12 # QP 求解器阻尼
```

Pink IK 从 URDF 构建 Pinocchio 模型，在 `gripper_base + localZ 0.1334 m` 添加虚拟 `model_tcp` 帧，通过 `FrameTask` + `PostureTask` + QP 求解差分运动，直接输出模型 TCP 空间的速度指令再积分回关节角，避免多初值搜索和显式逆变换。

### 固件 MOVE P（`firmware_move_p`）

`control_backend: firmware_move_p` 跳过主机 IK，由 `piper_sdk.EndPoseCtrl` 直接下发笛卡尔位姿。固件内部执行黑盒 IK。此模式下不生成 `host_ik` / `host_ik_reject` 日志。适合固件端 IK 已验证的场景或简单笛卡尔诊断。
