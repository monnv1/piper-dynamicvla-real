# 真机部署调试与实验记录

本文件既是调试手册，也是每次真机实验的记录模板。建议每次实验复制“实验记录模板”到文件末尾填写，并保留对应 `events.jsonl`。

## 快速故障定位

### Orbbec不可见

```bash
lsusb
python "$CONDA_PREFIX/lib/python3.10/site-packages/pyorbbecsdk/examples/beginner/01_hello_camera.py"
```

若 `lsusb`可见但SDK不可见，检查Orbbec udev规则并重新插拔。不要同时运行官方viewer和部署程序，同一设备只能被一个进程独占。

### D435i不可见

```bash
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
```

若两台相机单独正常、同时运行失败，执行 `lsusb -t` 检查是否共享同一个USB 3控制器或Hub。

### Piper没有反馈

```bash
ip -details link show can0
timeout 5 candump can0
```

要求接口为 `UP`、`ERROR-ACTIVE`、`bitrate 1000000`。机械臂需处于从机模式。diagnose不负责切换模式，避免无意发送配置命令。

### 模型加载失败

检查：

```bash
test -f /path/to/checkpoint/config.json
test -f /path/to/checkpoint/model.safetensors
```

确认checkpoint的 `input_features`包含：

```text
observation.images.opst_cam
observation.images.wrist_cam
observation.state
```

### 大量observation_skipped

通常是相机帧年龄超过 `camera_sync_tolerance_ms`。30 FPS相机的单帧周期约33 ms，因此默认容差为50 ms。先检查帧率和USB掉帧，不要直接长期放宽，否则时序视觉会失真。

### 大量stale chunk

检查GPU利用率、推理p95延迟和控制overrun。若推理超过0.8秒，20步chunk已全部失效；增加chunk年龄不能恢复动作的时序有效性。

## 日志事件

| 事件 | 含义 |
|---|---|
| `runtime_start` | 保存完整运行配置 |
| `devices_ready` | 双相机、Piper、模型均已准备 |
| `observation_skipped` | 本周期未得到同步双相机观测 |
| `action_chunk` | 新chunk完成，含推理时间、源index和完整动作 |
| `action` | 本周期调度动作，含raw、safe、state和是否执行 |
| `safety_reject` | 动作超出工作空间或格式非法 |
| `control_overrun` | 25 Hz循环超时 |
| `motion_enabled` | execute模式实际使能Piper |
| `runtime_stop` | 正常或异常退出时的LAAS统计 |

日志汇总：

```bash
python -m deploy.tools.summarize_log deploy/runs/<episode>/events.jsonl
```

进一步筛选：

```bash
rg 'safety_reject|control_overrun|observation_skipped' \
  deploy/runs/<episode>/events.jsonl
```

## Shadow验收清单

### Piper EndPose/FK坐标只读检测

保持机械臂静止且不要运行其他Piper进程：

```bash
python -m deploy.tools.verify_piper_frames \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --seconds 10
```

脚本只连接CAN、启用SDK本地FK计算并读取反馈；不会使能机械臂，也不会调用 `MotionCtrl_2`、`EndPoseCtrl`、`JointCtrl`或 `GripperCtrl`。把 `=== COPY EVERYTHING BELOW ===`后的JSON完整保存到实验记录。`end_pose_matches_fk_joint6=true`才表示当前 `link6 -> model_tcp`配置的前提成立；它不代表已经满足execute的其他安全条件。

### 低速移动到仿真训练起始构型

先执行只读dry-run并检查当前角度、目标角度和扫掠空间：

```bash
python -m deploy.tools.move_to_training_start \
  --config deploy/configs/piper_gemini_d435i.yaml
```

工具统一使用实机标定起始关节 `[0.0, 89.913, -80.913, 0.0, 58.398, 0.0] deg`，定义位于 `deploy/targets.py`。默认MOVE J速度5%，硬限制不超过10%，回位开始和完成时夹爪闭合至0 mm。实际运动必须清空扫掠空间、保持物理急停可达，并显式执行：

```bash
python -m deploy.tools.move_to_training_start \
  --config deploy/configs/piper_gemini_d435i.yaml \
  --speed-percent 5 \
  --execute \
  --confirm-motion
```

还必须在终端输入指定确认短语。该工具只有关节限位、ArmStatus、超时和到位误差联锁，**没有桌面、自碰撞、腕部相机、线缆或完整扫掠体碰撞检测**；它不能自行证明路径安全。禁止和 `deploy.run`或其他Piper程序同时运行。程序结束时终止轨迹并保持当前关节目标，电机继续使能以承受重力；绝不能在未机械支撑机械臂时调用 `DisableArm`或直接断电。

### 单关节低速微动

每次只测试一个关节，默认微动1°、速度3%，到位后停留并保持在最终角度，不自动返回。先dry-run，例如当前J2位于0°下限，只能先测+1°：

```bash
python -m deploy.tools.jog_piper_joint --joint 2 --delta-deg 1
```

实际运动：

```bash
python -m deploy.tools.jog_piper_joint \
  --joint 2 --delta-deg 1 \
  --execute --confirm-motion
```

当前代码允许0.2°到40°、速度1%到10%，默认仍为1°和3%。超过5°已经不属于“微动”，必须先检查dry-run的完整FK路径和实体扫掠空间。程序检查关节限位、近似FK路径workspace和ArmStatus，但仍不具备完整碰撞检测。每次只运行一个命令并观察实体方向。测试结束后电机保持使能并锁定最终位置，不自动返回，也不调用 `DisableArm`。

### 移动到已有标定零位

六关节零位为 `[0,0,0,0,0,0]°`。先dry-run：

```bash
python -m deploy.tools.move_to_joint_zero
```

实际低速回零：

```bash
python -m deploy.tools.move_to_joint_zero \
  --speed-percent 3 \
  --execute --confirm-motion
```

按提示输入 `MOVE_PIPER_SLOWLY_TO_CALIBRATED_ZERO`。该工具仅移动到现有标定零位，不调用 `JointConfig(..., 0xAE)`，不会修改编码器零点。零位不是DynamicVLA训练起点。程序结束后保持使能。

- [ ] 双相机连续运行30分钟无断流。
- [ ] Piper反馈时间戳连续，反馈频率稳定。
- [ ] D435i固定第三视角图像确实对应 `opst_cam`。
- [ ] Gemini 305腕部图像确实对应 `wrist_cam`。
- [ ] RGB颜色正确，无红蓝通道交换。
- [ ] checkpoint为6维state时，模型输入是 `[x,y,z,rx,ry,rz]`。
- [ ] 模型action为 `[x,y,z,rx,ry,rz,gripper]`。
- [ ] Piper末端位置单位已确认是米。
- [ ] Piper Euler顺序和模型均为外旋XYZ。
- [ ] SDK EndPose与 `GetFK("feedback")[-1]`一致，确认其为link6而非固件补偿后的工具TCP。
- [ ] YAML中的 `T_sdk_model_tcp = Rz(pi) + local Z 0.1334 m`与实体/固件配置一致。
- [ ] Delta action已由模型配置自动转换为绝对目标。
- [ ] 模型夹爪 `+1=打开、-1=闭合`，Piper开闭方向已实机确认。
- [ ] `gripper_min_m/gripper_max_m`与实体夹爪行程一致。
- [ ] 实测工作空间已写入YAML。
- [ ] shadow中无无法解释的 `safety_reject`。
- [ ] 推理p95延迟小于chunk覆盖时间。
- [ ] LAAS跳过步数与推理延迟/40 ms一致。
- [ ] 断开任意相机后运行时能停止而非继续生成新动作。
- [ ] 断开CAN反馈后运行时能因stale feedback停止。
- [ ] 急停按钮在操作人员可触及位置。

## 首次execute限制

首次仅使用专用配置：

```bash
python -m deploy.run \
  --config deploy/configs/piper_gemini_d435i_first_execute.yaml \
  --confirm-motion
```

当前专用配置为Piper速度3%，model TCP单周期平移/旋转分别限制为3 mm和0.02 rad，夹爪关闭，且15秒后自动结束命令并保持当前关节位置。这已经不是首轮最低档，运行时必须持续观察实体机械臂。

- 工作区内不放置人、动态物体或易碎物。
- 机械臂使用最低可行速度，配置默认10%。
- 初始任务使用固定目标和小位移。
- 操作人员保持急停可达。
- 首次只开放末端位姿，不测试夹爪闭合。
- 每次运行不超过数秒，立即复查日志。

## 实验记录模板

### YYYY-MM-DD / 实验编号

基本信息：

```text
操作者：
Git commit：
模式：diagnose / shadow / execute
配置文件：
episode目录：
模型路径：
模型epoch/hash：
任务指令：
```

硬件：

```text
Piper固件：
Piper dh_is_offset：
CAN接口/适配器：
Gemini序列号/固件/USB端口：
D435i序列号/固件/USB端口：
GPU型号：
```

校准与安全：

```text
TCP定义：
Gemini外参版本：
D435i手眼标定版本：
工作空间min/max：
最大单步平移/旋转：
速度百分比：
```

结果：

```text
运行时长：
平均/p95/max推理延迟：
平均/max LAAS跳步：
observation_skipped数量：
safety_reject数量及原因：
control_overrun数量：
是否发生设备断流：
是否发生非预期运动：
结论：通过 / 不通过
```

问题与下一步：

```text
现象：
复现步骤：
相关日志行：
初步原因：
修改内容：
回归验证：
```
