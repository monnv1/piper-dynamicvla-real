import numpy as np
import pytest

from deploy.tools.move_to_training_start import (
    TRAINING_START_DEG,
    TRAINING_START_GRIPPER_M,
    TRAINING_START_RAD,
    move_gripper_and_wait,
    send_gripper_target,
    set_gripper_zero_at_closed_position,
    validate_joint_values,
    validate_tcp_path,
)


def test_training_target_matches_calibrated_physical_configuration() -> None:
    np.testing.assert_allclose(
        TRAINING_START_DEG,
        [0.0, 70.913, -60.913, 0.0, 58.398, 0.0],
        atol=1e-9,
    )
    np.testing.assert_allclose(TRAINING_START_RAD, np.radians(TRAINING_START_DEG))
    assert TRAINING_START_GRIPPER_M == pytest.approx(0.07)
    validate_joint_values(TRAINING_START_DEG, "target", margin_deg=0.5)


def test_training_start_gripper_is_sent_fully_open() -> None:
    class FakePiper:
        def __init__(self) -> None:
            self.command = None

        def GripperCtrl(self, *command) -> None:
            self.command = command

    piper = FakePiper()
    send_gripper_target(piper, TRAINING_START_GRIPPER_M)
    assert piper.command == (70000, 1000, 0x01, 0x00)


def test_training_start_gripper_clears_enables_and_waits_for_feedback(
    monkeypatch,
) -> None:
    class FocStatus:
        voltage_too_low = False
        motor_overheating = False
        driver_overcurrent = False
        driver_overheating = False
        sensor_status = False
        driver_error_status = False
        driver_enable_status = True
        homing_status = True

    class GripperState:
        grippers_angle = 0
        foc_status = FocStatus()

    class Message:
        Hz = 100.0
        gripper_state = GripperState()

    class FakePiper:
        def __init__(self) -> None:
            self.commands = []
            self.feedback = Message()

        def GripperCtrl(self, angle, effort, code, set_zero) -> None:
            self.commands.append((angle, effort, code, set_zero))
            if code == 0x01:
                self.feedback.gripper_state.grippers_angle = angle

        def GetArmGripperMsgs(self):
            return self.feedback

    monkeypatch.setattr("deploy.tools.move_to_training_start.time.sleep", lambda _: None)
    piper = FakePiper()
    reached = move_gripper_and_wait(piper, TRAINING_START_GRIPPER_M)

    assert reached == pytest.approx(TRAINING_START_GRIPPER_M)
    assert piper.commands[0][2] == 0x02
    assert piper.commands[1][2] == 0x03
    assert piper.commands[-1] == (70000, 1000, 0x01, 0x00)


def test_gripper_zero_uses_official_disable_then_set_zero_sequence(
    monkeypatch,
) -> None:
    class FocStatus:
        homing_status = False

    class GripperState:
        grippers_angle = 0
        foc_status = FocStatus()

    class Message:
        Hz = 100.0
        gripper_state = GripperState()

    class FakePiper:
        def __init__(self) -> None:
            self.commands = []
            self.feedback = Message()

        def GripperCtrl(self, angle, effort, code, set_zero) -> None:
            self.commands.append((angle, effort, code, set_zero))

        def GetArmGripperMsgs(self):
            return self.feedback

    monkeypatch.setattr("builtins.input", lambda _: "GRIPPER_IS_PHYSICALLY_CLOSED_SET_ZERO")
    monkeypatch.setattr("deploy.tools.move_to_training_start.time.sleep", lambda _: None)
    piper = FakePiper()
    set_gripper_zero_at_closed_position(piper)

    assert piper.commands == [
        (0, 1000, 0x00, 0x00),
        (0, 1000, 0x00, 0xAE),
    ]


def test_joint_limit_validation_rejects_out_of_range_target() -> None:
    invalid = TRAINING_START_DEG.copy()
    invalid[4] = 75.0
    with pytest.raises(RuntimeError, match="J5"):
        validate_joint_values(invalid, "target")


def test_tcp_path_validation_rejects_workspace_exit() -> None:
    path = np.asarray([[0.3, 0.0, 0.2], [0.1, 0.0, 0.2]])
    with pytest.raises(RuntimeError, match="sample 1"):
        validate_tcp_path(
            path,
            np.asarray([0.15, -0.35, 0.05]),
            np.asarray([0.60, 0.35, 0.55]),
        )
