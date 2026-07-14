"""诊断和切换 Piper 机械臂控制模式。

物理示教按钮（夹爪上）是唯一能开启重力补偿的方式：
  按一下 → 灯亮 → ctrl_mode=TEACHING(0x2), teach_status=START_RECORDING → 重力补偿生效
  再按一下 → 灯灭 → 抱闸锁死，关节无法拖动

软件命令 MotionCtrl_1(grag_teach_ctrl=0x01) 只能让 ctrl_mode→TEACHING_MODE，
但 teach_status 仍为 DISABLED，重力补偿不会生效。

本脚本只做两件事：
  --can can0         读取当前状态
  --can can0 --reset 恢复到 STANDBY 模式并使能电机
"""
from __future__ import annotations

import argparse
import time


def read_status(piper):
    wrapper = piper.GetArmStatus()
    s = getattr(wrapper, "arm_status", wrapper)
    return {
        "ctrl_mode": s.ctrl_mode,
        "arm_status": s.arm_status,
        "mode_feed": s.mode_feed,
        "teach_status": s.teach_status,
        "motion_status": s.motion_status,
    }


STATUS_HELP = {
    "ctrl_mode": {
        0x00: "STANDBY(0x0) 待机",
        0x01: "CAN_CTRL(0x1) CAN指令控制",
        0x02: "TEACHING(0x2) 示教模式",
        0x06: "LINKAGE_TEACHING_INPUT(0x6) 联动示教输入",
    },
    "arm_status": {
        0x00: "NORMAL 正常",
        0x01: "EMERGENCY_STOP 急停",
        0x06: "JOINT_BRAKE_CLOSED 抱闸锁死",
        0x08: "TEACH_OVERSPEED 拖动超速",
    },
    "mode_feed": {
        0x00: "MOVE_P",
        0x01: "MOVE_J",
    },
    "teach_status": {
        0x00: "DISABLED 重力补偿关闭",
        0x01: "START_RECORDING 重力补偿开启",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断/切换 Piper 控制模式")
    parser.add_argument("--can", default="can0", help="CAN 接口 (默认: can0)")
    parser.add_argument(
        "--reset", action="store_true",
        help="恢复到 STANDBY 模式并重新使能电机",
    )
    args = parser.parse_args()

    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(
        can_name=args.can,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )
    piper.ConnectPort(piper_init=False, start_thread=True)
    time.sleep(1)

    s = read_status(piper)
    enable = piper.GetArmEnableStatus()

    print(f"机械臂 [{args.can}] 当前状态:")
    print(f"  ctrl_mode:    0x{int(s['ctrl_mode']):02X}  ({STATUS_HELP['ctrl_mode'].get(int(s['ctrl_mode']), '未知')})")
    print(f"  arm_status:   0x{int(s['arm_status']):02X}  ({STATUS_HELP['arm_status'].get(int(s['arm_status']), '未知')})")
    print(f"  mode_feed:    0x{int(s['mode_feed']):02X}  ({STATUS_HELP['mode_feed'].get(int(s['mode_feed']), '未知')})")
    print(f"  teach_status: 0x{int(s['teach_status']):02X}  ({STATUS_HELP['teach_status'].get(int(s['teach_status']), '未知')})")
    print(f"  电机使能: {enable}")

    if int(s["teach_status"]) == 0x01:
        print("\n✅ 重力补偿已开启，可自由拖拽")
    elif int(s["ctrl_mode"]) == 0x02 and int(s["teach_status"]) == 0x00:
        print("\n⚠️  ctrl_mode=TEACHING 但 teach_status=DISABLED")
        print("   重力补偿未生效。请按一下夹爪上的物理示教按钮。")
    elif int(s["ctrl_mode"]) == 0x00 and int(s["arm_status"]) == 0x06:
        print("\n🔒 抱闸锁死 (arm_status=0x06 JOINT_BRAKE_CLOSED)")
        print("   需要按示教按钮进入重力补偿，或用 --reset 进入 CAN 控制模式")

    if args.reset:
        print("\n--- 执行重置 ---")
        print("1. 发送 reset (MotionCtrl_1 emergency_stop=0x02)...")
        piper.MotionCtrl_1(0x02, 0, 0)
        time.sleep(0.3)

        print("2. 使能电机...")
        piper.EnablePiper()
        time.sleep(0.5)

        s = read_status(piper)
        enable = piper.GetArmEnableStatus()
        print(f"   结果: ctrl_mode=0x{int(s['ctrl_mode']):02X}  enable={enable}")
        print("   电机已使能但无重力补偿。如需拖拽请按示教按钮。")


if __name__ == "__main__":
    main()
