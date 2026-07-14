from deploy.config import load_config


def test_first_execute_config_is_bounded_and_slow() -> None:
    config = load_config("deploy/configs/piper_gemini_d435i_first_execute.yaml")
    assert config.runtime.mode == "execute"
    assert config.runtime.max_execute_seconds == 30.0
    assert config.runtime.return_to_training_start_on_normal_exit is True
    assert config.runtime.return_speed_percent == 5
    assert config.robot.auto_enable is True
    assert config.robot.command_gripper is True
    assert config.robot.command_speed_percent == 5
    assert config.safety.max_translation_step_m == 0.01
    assert config.safety.max_rotation_step_rad == 0.08
