from deploy.tools.print_gripper_state import format_gripper_state


def test_format_gripper_state_is_single_line_and_exposes_status() -> None:
    class Foc:
        voltage_too_low = False
        motor_overheating = False
        driver_overcurrent = False
        driver_overheating = False
        sensor_status = False
        driver_error_status = False
        driver_enable_status = True
        homing_status = False

    class State:
        grippers_angle = 70000
        grippers_effort = 1000
        status_code = 0x40
        foc_status = Foc()

    class Feedback:
        Hz = 200.0
        gripper_state = State()

    output = format_gripper_state(Feedback())

    assert "\n" not in output
    assert "position=70.000mm" in output
    assert "status=0x40" in output
    assert "enabled=True" in output
    assert "homed=False" in output
    assert "faults=none" in output
    assert "feedback_hz=200.0" in output
