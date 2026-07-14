import numpy as np
import pytest

from deploy.kinematics.piper_ik import HostIKError, PiperHostIK


def make_ik(max_joint_step_deg: float = 5.0) -> tuple[PiperHostIK, object]:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(1)
    ik = PiperHostIK(
        fk=fk,
        position_tolerance_m=0.0005,
        rotation_tolerance_rad=0.005,
        max_joint_step_deg=max_joint_step_deg,
        min_joint_limit_margin_deg=0.2,
        max_nfev=80,
    )
    return ik, fk


def target_from_joints(fk, joints_deg: list[float]):
    joints_rad = np.radians(joints_deg)
    pose = np.asarray(fk.CalFK(joints_rad)[-1], dtype=np.float64)
    return pose[:3] / 1000.0, np.radians(pose[3:])


def test_host_ik_recovers_a_nearby_valid_joint_solution() -> None:
    ik, fk = make_ik()
    target_deg = np.asarray([0.0, 50.0, -50.0, 0.0, 45.0, 0.0])
    position, euler = target_from_joints(fk, target_deg.tolist())
    current_deg = target_deg + np.asarray([1.0, -1.0, 1.0, 0.0, -1.0, 0.0])

    result = ik.solve(position, euler, np.radians(current_deg))

    np.testing.assert_allclose(
        result["selected_joint_degrees"], target_deg, atol=0.05
    )
    assert result["selected"]["position_error_m"] < 0.0005
    assert result["selected"]["rotation_error_rad"] < 0.005
    assert result["selected"]["maximum_step_from_current_deg"] <= 5.0


def test_host_ik_scales_a_solution_requiring_a_large_joint_jump() -> None:
    ik, fk = make_ik(max_joint_step_deg=0.1)
    target_deg = np.asarray([0.0, 50.0, -50.0, 0.0, 45.0, 0.0])
    position, euler = target_from_joints(fk, target_deg.tolist())
    current_deg = target_deg + np.asarray([2.0, -2.0, 2.0, 0.0, -2.0, 0.0])

    result = ik.solve(position, euler, np.radians(current_deg))

    assert result["joint_step_limited"]
    assert 0.0 < result["joint_step_scale"] < 1.0
    assert result["requested_max_joint_step_deg"] > 0.1
    assert result["commanded_max_joint_step_deg"] == pytest.approx(0.1)
    np.testing.assert_allclose(
        result["ik_solution_joint_degrees"], target_deg, atol=0.05
    )
    assert np.max(
        np.abs(np.asarray(result["selected_joint_degrees"]) - current_deg)
    ) <= 0.1 + 1e-9


def test_host_ik_still_rejects_an_unreachable_cartesian_target() -> None:
    ik, _ = make_ik()

    with pytest.raises(HostIKError) as raised:
        ik.solve(
            np.asarray([10.0, 10.0, 10.0]),
            np.zeros(3),
            np.radians([0.0, 50.0, -50.0, 0.0, 45.0, 0.0]),
        )

    diagnostics = raised.value.diagnostics
    assert diagnostics["candidates"]
    assert not any(candidate["accepted"] for candidate in diagnostics["candidates"])



def test_host_ik_projects_pose_when_exact_solution_is_near_joint_limit() -> None:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(1)
    ik = PiperHostIK(
        fk=fk,
        position_tolerance_m=0.0005,
        rotation_tolerance_rad=0.005,
        max_joint_step_deg=20.0,
        min_joint_limit_margin_deg=0.2,
        max_nfev=100,
        allow_pose_projection=True,
        projection_joint_limit_margin_deg=2.0,
        projection_max_position_error_m=0.01,
        projection_max_rotation_error_rad=0.2,
        projection_position_weight=1.0,
        projection_rotation_weight=0.25,
    )
    target_deg = np.asarray([0.0, 50.0, -50.0, 0.0, 69.5, 0.0])
    position, euler = target_from_joints(fk, target_deg.tolist())

    result = ik.solve(
        position,
        euler,
        np.radians([0.0, 50.0, -50.0, 0.0, 60.0, 0.0]),
    )

    assert result["exact_solution_available"]
    assert result["pose_projected"]
    assert result["selected"]["rejection_reason"] is None
    assert result["pose_projection_reason"] == "exact_solution_near_joint_limit"
    assert result["selected"]["minimum_limit_margin_deg"] >= 2.0 - 1e-6
    assert result["ik_solution_joint_degrees"][4] <= 68.0 + 1e-6
    assert result["commanded_position_error_m"] < 0.01
    assert result["commanded_rotation_error_rad"] < 0.2
    assert result["projection_candidates"]


def test_host_ik_rejects_projection_when_nearest_pose_is_too_far() -> None:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(1)
    ik = PiperHostIK(
        fk=fk,
        position_tolerance_m=0.0005,
        rotation_tolerance_rad=0.005,
        max_joint_step_deg=20.0,
        min_joint_limit_margin_deg=0.2,
        max_nfev=80,
        allow_pose_projection=True,
        projection_joint_limit_margin_deg=2.0,
        projection_max_position_error_m=0.000001,
        projection_max_rotation_error_rad=0.000001,
        projection_position_weight=1.0,
        projection_rotation_weight=0.25,
    )
    target_deg = np.asarray([0.0, 50.0, -50.0, 0.0, 69.5, 0.0])
    position, euler = target_from_joints(fk, target_deg.tolist())

    with pytest.raises(HostIKError) as raised:
        ik.solve(
            position,
            euler,
            np.radians([0.0, 50.0, -50.0, 0.0, 60.0, 0.0]),
        )

    assert raised.value.diagnostics["pose_projection_enabled"]
    assert raised.value.diagnostics["candidates"]



def test_differential_ik_moves_toward_a_small_cartesian_target() -> None:
    from piper_sdk import C_PiperForwardKinematics

    from deploy.kinematics.piper_ik import PiperDifferentialIK

    fk = C_PiperForwardKinematics(1)
    ik = PiperDifferentialIK(
        fk=fk,
        max_joint_step_deg=1.0,
        min_joint_limit_margin_deg=0.0,
        lambda_val=0.01,
        finite_difference_eps_rad=1e-4,
    )
    current_rad = np.radians([0.0, 70.0, -60.0, 0.0, 58.0, 0.0])
    start_position, start_euler = target_from_joints(
        fk, np.degrees(current_rad).tolist()
    )
    target_position = start_position + np.asarray([0.001, 0.0, 0.0])

    result = ik.solve(target_position, start_euler, current_rad)

    assert result["solver"] == "numerical_dls_differential_ik"
    assert result["commanded_position_error_m"] < 0.001
    assert result["commanded_max_joint_step_deg"] <= 1.0 + 1e-9
    assert not result["joint_limit_clipped"]



def test_pink_ik_moves_toward_a_small_cartesian_target() -> None:
    from scipy.spatial.transform import Rotation

    from deploy.kinematics.piper_pink_ik import PiperPinkIK

    ik = PiperPinkIK(
        "/data/repos/DynamicVLA/simulations/robots/PIPER/piper_description.urdf",
        max_joint_step_deg=1.0,
    )
    current_rad = np.radians([0.0, 70.0, -60.0, 0.0, 58.0, 0.0])
    configuration = ik._configuration(current_rad)
    current_transform = configuration.get_transform_frame_to_world("model_tcp")
    target_position = current_transform.translation + np.asarray([0.001, 0.0, 0.0])
    target_euler = Rotation.from_matrix(current_transform.rotation).as_euler("xyz")

    result = ik.solve(target_position, target_euler, current_rad)

    assert result["solver"] == "pink_pinocchio_qp"
    assert result["commanded_position_error_m"] < 0.001
    assert result["commanded_max_joint_step_deg"] <= 1.0 + 1e-9
    assert not result["joint_limit_clipped"]
