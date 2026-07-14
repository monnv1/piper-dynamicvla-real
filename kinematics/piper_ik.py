from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


JOINT_LOWER_DEG = np.asarray([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0])
JOINT_UPPER_DEG = np.asarray([150.0, 180.0, 0.0, 100.0, 70.0, 120.0])


class HostIKError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class IKCandidate:
    seed_degrees: list[float]
    joint_degrees: list[float]
    position_error_m: float
    rotation_error_rad: float
    minimum_limit_margin_deg: float
    maximum_step_from_current_deg: float
    cost: float
    nfev: int
    solver_success: bool
    accepted: bool
    rejection_reason: str | None


class PiperHostIK:
    """Bounded multi-seed IK using Piper's official SDK forward kinematics."""

    def __init__(
        self,
        fk,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        max_joint_step_deg: float,
        min_joint_limit_margin_deg: float,
        max_nfev: int,
        allow_pose_projection: bool = False,
        projection_joint_limit_margin_deg: float = 2.0,
        projection_max_position_error_m: float = 0.003,
        projection_max_rotation_error_rad: float = 0.08,
        projection_position_weight: float = 1.0,
        projection_rotation_weight: float = 0.25,
    ) -> None:
        self.fk = fk
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.min_joint_limit_margin_deg = float(min_joint_limit_margin_deg)
        self.max_nfev = int(max_nfev)
        self.allow_pose_projection = bool(allow_pose_projection)
        self.projection_joint_limit_margin_deg = float(projection_joint_limit_margin_deg)
        self.projection_max_position_error_m = float(projection_max_position_error_m)
        self.projection_max_rotation_error_rad = float(projection_max_rotation_error_rad)
        self.projection_position_weight = float(projection_position_weight)
        self.projection_rotation_weight = float(projection_rotation_weight)
        self.lower_deg = JOINT_LOWER_DEG + self.min_joint_limit_margin_deg
        self.upper_deg = JOINT_UPPER_DEG - self.min_joint_limit_margin_deg
        self.lower_rad = np.radians(self.lower_deg)
        self.upper_rad = np.radians(self.upper_deg)
        self.projection_lower_deg = (
            JOINT_LOWER_DEG + self.projection_joint_limit_margin_deg
        )
        self.projection_upper_deg = (
            JOINT_UPPER_DEG - self.projection_joint_limit_margin_deg
        )
        self.projection_lower_rad = np.radians(self.projection_lower_deg)
        self.projection_upper_rad = np.radians(self.projection_upper_deg)
        self.last_solution_rad: np.ndarray | None = None
        self.last_projection_candidates: list[IKCandidate] = []
        self.last_pose_projection_reason: str | None = None

    def _fk_pose(self, joints_rad: np.ndarray) -> tuple[np.ndarray, Rotation]:
        pose = np.asarray(self.fk.CalFK(joints_rad.tolist())[-1], dtype=np.float64)
        return pose[:3] / 1000.0, Rotation.from_euler(
            "xyz", np.radians(pose[3:])
        )

    def _errors(
        self,
        joints_rad: np.ndarray,
        target_position_m: np.ndarray,
        target_rotation: Rotation,
    ) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self._fk_pose(joints_rad)
        position_error = position - target_position_m
        rotation_error = (target_rotation * rotation.inv()).as_rotvec()
        return position_error, rotation_error

    def _residual(
        self,
        joints_rad: np.ndarray,
        target_position_m: np.ndarray,
        target_rotation: Rotation,
    ) -> np.ndarray:
        position_error, rotation_error = self._errors(
            joints_rad, target_position_m, target_rotation
        )
        return np.concatenate(
            (
                position_error / self.position_tolerance_m,
                rotation_error / self.rotation_tolerance_rad,
            )
        )

    def _projection_residual(
        self,
        joints_rad: np.ndarray,
        target_position_m: np.ndarray,
        target_rotation: Rotation,
        current_joint_rad: np.ndarray,
    ) -> np.ndarray:
        position_error, rotation_error = self._errors(
            joints_rad, target_position_m, target_rotation
        )
        continuity_error = (
            np.degrees(joints_rad - current_joint_rad)
            / (JOINT_UPPER_DEG - JOINT_LOWER_DEG)
        )
        return np.concatenate(
            (
                self.projection_position_weight
                * position_error
                / self.projection_max_position_error_m,
                self.projection_rotation_weight
                * rotation_error
                / self.projection_max_rotation_error_rad,
                0.05 * continuity_error,
            )
        )

    def _seeds(self, current_rad: np.ndarray) -> list[np.ndarray]:
        seeds: list[np.ndarray] = []
        if self.last_solution_rad is not None:
            seeds.append(self.last_solution_rad)
        seeds.append(current_rad)
        templates_deg = [
            [0.0, 50.0, -50.0, 0.0, 45.0, 0.0],
            [0.0, 90.0, -90.0, 0.0, 0.0, 0.0],
            [0.0, 90.0, -90.0, 0.0, 55.0, 0.0],
            [0.0, 90.0, -90.0, 0.0, -55.0, 0.0],
            [0.0, 130.0, -130.0, 0.0, 0.0, 0.0],
        ]
        for template in templates_deg:
            seed = np.radians(template)
            seed[0] = current_rad[0]
            seeds.append(seed)

        unique: list[np.ndarray] = []
        for seed in seeds:
            clipped = np.clip(seed, self.lower_rad, self.upper_rad)
            if not any(np.linalg.norm(clipped - item) < 1e-5 for item in unique):
                unique.append(clipped)
        return unique

    def solve(
        self,
        target_position_m: np.ndarray,
        target_euler_xyz_rad: np.ndarray,
        current_joint_rad: np.ndarray,
    ) -> dict[str, object]:
        started = time.monotonic()
        target_position_m = np.asarray(target_position_m, dtype=np.float64)
        target_euler_xyz_rad = np.asarray(target_euler_xyz_rad, dtype=np.float64)
        current_joint_rad = np.asarray(current_joint_rad, dtype=np.float64)
        if target_position_m.shape != (3,) or target_euler_xyz_rad.shape != (3,):
            raise ValueError("IK target must contain 3D position and Euler rotation")
        if current_joint_rad.shape != (6,):
            raise ValueError("Current Piper joints must contain six angles")
        if not (
            np.isfinite(target_position_m).all()
            and np.isfinite(target_euler_xyz_rad).all()
            and np.isfinite(current_joint_rad).all()
        ):
            raise ValueError("IK input contains NaN or Inf")

        target_rotation = Rotation.from_euler("xyz", target_euler_xyz_rad)
        current_deg = np.degrees(current_joint_rad)
        candidates: list[IKCandidate] = []
        accepted_solutions: list[tuple[float, np.ndarray, IKCandidate]] = []

        primary_seed_count = 1
        if self.last_solution_rad is not None and np.linalg.norm(
            self.last_solution_rad - current_joint_rad
        ) >= 1e-5:
            primary_seed_count = 2
        for seed_index, seed in enumerate(self._seeds(current_joint_rad)):
            result = least_squares(
                self._residual,
                seed,
                args=(target_position_m, target_rotation),
                bounds=(self.lower_rad, self.upper_rad),
                method="trf",
                x_scale="jac",
                max_nfev=self.max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            solution_rad = np.asarray(result.x, dtype=np.float64)
            solution_deg = np.degrees(solution_rad)
            position_error, rotation_error = self._errors(
                solution_rad, target_position_m, target_rotation
            )
            position_error_m = float(np.linalg.norm(position_error))
            rotation_error_rad = float(np.linalg.norm(rotation_error))
            limit_margin = float(
                np.min(
                    np.minimum(
                        solution_deg - JOINT_LOWER_DEG,
                        JOINT_UPPER_DEG - solution_deg,
                    )
                )
            )
            max_step = float(np.max(np.abs(solution_deg - current_deg)))
            reason = None
            if position_error_m > self.position_tolerance_m:
                reason = "position_error"
            elif rotation_error_rad > self.rotation_tolerance_rad:
                reason = "rotation_error"
            elif limit_margin < self.min_joint_limit_margin_deg:
                reason = "joint_limit_margin"
            accepted = reason is None
            normalized_step = np.linalg.norm(
                (solution_deg - current_deg) / (JOINT_UPPER_DEG - JOINT_LOWER_DEG)
            )
            score = float(
                normalized_step
                + 0.02 / max(limit_margin, 1e-6)
                + 10.0 * position_error_m
                + rotation_error_rad
            )
            candidate = IKCandidate(
                seed_degrees=np.degrees(seed).tolist(),
                joint_degrees=solution_deg.tolist(),
                position_error_m=position_error_m,
                rotation_error_rad=rotation_error_rad,
                minimum_limit_margin_deg=limit_margin,
                maximum_step_from_current_deg=max_step,
                cost=float(result.cost),
                nfev=int(result.nfev),
                solver_success=bool(result.success),
                accepted=accepted,
                rejection_reason=reason,
            )
            candidates.append(candidate)
            if accepted:
                accepted_solutions.append((score, solution_rad, candidate))
                if seed_index < primary_seed_count:
                    break

        diagnostics: dict[str, object] = {
            "solver": "scipy_least_squares_bounded_multiseed",
            "target_sdk_link6": {
                "position_m": target_position_m,
                "euler_xyz_rad": target_euler_xyz_rad,
                "euler_xyz_deg": np.degrees(target_euler_xyz_rad),
            },
            "current_joint_degrees": current_deg,
            "joint_lower_degrees": JOINT_LOWER_DEG,
            "joint_upper_degrees": JOINT_UPPER_DEG,
            "position_tolerance_m": self.position_tolerance_m,
            "rotation_tolerance_rad": self.rotation_tolerance_rad,
            "max_joint_step_deg": self.max_joint_step_deg,
            "pose_projection_enabled": self.allow_pose_projection,
            "projection_joint_limit_margin_deg": self.projection_joint_limit_margin_deg,
            "projection_max_position_error_m": self.projection_max_position_error_m,
            "projection_max_rotation_error_rad": self.projection_max_rotation_error_rad,
            "candidates": [asdict(candidate) for candidate in candidates],
            "solve_seconds": time.monotonic() - started,
        }
        projected_solution = None
        exact_solution_available = bool(accepted_solutions)
        if accepted_solutions:
            accepted_solutions.sort(key=lambda item: item[0])
            exact_score, exact_solution_rad, exact_selected = accepted_solutions[0]
            if (
                self.allow_pose_projection
                and exact_selected.minimum_limit_margin_deg
                < self.projection_joint_limit_margin_deg
            ):
                projected_solution = self._solve_projection(
                    target_position_m,
                    target_rotation,
                    current_joint_rad,
                    "exact_solution_near_joint_limit",
                )
                diagnostics["projection_candidates"] = [
                    asdict(candidate)
                    for candidate in self.last_projection_candidates
                ]
                if projected_solution is None:
                    raise HostIKError("No safe host IK solution", diagnostics)
            if projected_solution is None:
                score, solution_rad, selected = (
                    exact_score,
                    exact_solution_rad,
                    exact_selected,
                )
            else:
                score, solution_rad, selected = projected_solution
        elif self.allow_pose_projection:
            projected_solution = self._solve_projection(
                target_position_m,
                target_rotation,
                current_joint_rad,
                "no_exact_safe_solution",
            )
            diagnostics["projection_candidates"] = [
                asdict(candidate) for candidate in self.last_projection_candidates
            ]
            if projected_solution is None:
                raise HostIKError("No safe host IK solution", diagnostics)
            score, solution_rad, selected = projected_solution
        else:
            raise HostIKError("No safe host IK solution", diagnostics)

        diagnostics["projection_candidates"] = [
            asdict(candidate) for candidate in self.last_projection_candidates
        ]

        # The exact IK solution may be farther from measured feedback than one
        # control cycle permits. Preserve the coordinated joint-space direction
        # by scaling the complete six-axis delta instead of clipping axes
        # independently or rejecting an otherwise valid Cartesian solution.
        solution_delta_rad = solution_rad - current_joint_rad
        requested_max_step_deg = float(
            np.max(np.abs(np.degrees(solution_delta_rad)))
        )
        if requested_max_step_deg > self.max_joint_step_deg:
            joint_step_scale = self.max_joint_step_deg / requested_max_step_deg
        else:
            joint_step_scale = 1.0
        command_rad = current_joint_rad + joint_step_scale * solution_delta_rad
        command_position_error, command_rotation_error = self._errors(
            command_rad, target_position_m, target_rotation
        )
        command_deg = np.degrees(command_rad)
        command_limit_margin_deg = float(
            np.min(
                np.minimum(
                    command_deg - JOINT_LOWER_DEG,
                    JOINT_UPPER_DEG - command_deg,
                )
            )
        )

        # Warm-start from what was actually prepared for JointCtrl, not from an
        # exact IK endpoint that the step limiter deliberately did not send.
        self.last_solution_rad = command_rad.copy()
        diagnostics.update(
            {
                "selected_joint_degrees": command_deg,
                "selected_joint_radians": command_rad,
                "ik_solution_joint_degrees": np.degrees(solution_rad),
                "ik_solution_joint_radians": solution_rad,
                "selected_score": score,
                "selected": asdict(selected),
                "exact_solution_available": exact_solution_available,
                "pose_projected": projected_solution is not None,
                "pose_projection_reason": (
                    self.last_pose_projection_reason
                    if projected_solution is not None
                    else None
                ),
                "joint_step_limited": joint_step_scale < 1.0,
                "joint_step_scale": float(joint_step_scale),
                "requested_max_joint_step_deg": requested_max_step_deg,
                "commanded_max_joint_step_deg": float(
                    np.max(np.abs(command_deg - current_deg))
                ),
                "commanded_minimum_limit_margin_deg": command_limit_margin_deg,
                "commanded_position_error_m": float(
                    np.linalg.norm(command_position_error)
                ),
                "commanded_rotation_error_rad": float(
                    np.linalg.norm(command_rotation_error)
                ),
            }
        )
        return diagnostics

    def _solve_projection(
        self,
        target_position_m: np.ndarray,
        target_rotation: Rotation,
        current_joint_rad: np.ndarray,
        reason: str,
    ) -> tuple[float, np.ndarray, IKCandidate] | None:
        self.last_projection_candidates = []
        self.last_pose_projection_reason = None
        projected: list[tuple[float, np.ndarray, IKCandidate]] = []
        current_deg = np.degrees(current_joint_rad)
        for seed in self._seeds(current_joint_rad):
            seed = np.clip(seed, self.projection_lower_rad, self.projection_upper_rad)
            result = least_squares(
                self._projection_residual,
                seed,
                args=(target_position_m, target_rotation, current_joint_rad),
                bounds=(self.projection_lower_rad, self.projection_upper_rad),
                method="trf",
                x_scale="jac",
                max_nfev=self.max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            solution_rad = np.asarray(result.x, dtype=np.float64)
            solution_deg = np.degrees(solution_rad)
            position_error, rotation_error = self._errors(
                solution_rad, target_position_m, target_rotation
            )
            position_error_m = float(np.linalg.norm(position_error))
            rotation_error_rad = float(np.linalg.norm(rotation_error))
            limit_margin = float(
                np.min(
                    np.minimum(
                        solution_deg - JOINT_LOWER_DEG,
                        JOINT_UPPER_DEG - solution_deg,
                    )
                )
            )
            max_step = float(np.max(np.abs(solution_deg - current_deg)))
            if position_error_m > self.projection_max_position_error_m:
                rejection_reason = "projection_position_error"
            elif rotation_error_rad > self.projection_max_rotation_error_rad:
                rejection_reason = "projection_rotation_error"
            elif limit_margin < self.projection_joint_limit_margin_deg:
                rejection_reason = "projection_joint_limit_margin"
            else:
                rejection_reason = None
            accepted = rejection_reason is None
            normalized_step = np.linalg.norm(
                (solution_deg - current_deg) / (JOINT_UPPER_DEG - JOINT_LOWER_DEG)
            )
            score = float(
                normalized_step
                + 0.05 / max(limit_margin, 1e-6)
                + 10.0 * position_error_m
                + self.projection_rotation_weight * rotation_error_rad
            )
            candidate = IKCandidate(
                seed_degrees=np.degrees(seed).tolist(),
                joint_degrees=solution_deg.tolist(),
                position_error_m=position_error_m,
                rotation_error_rad=rotation_error_rad,
                minimum_limit_margin_deg=limit_margin,
                maximum_step_from_current_deg=max_step,
                cost=float(result.cost),
                nfev=int(result.nfev),
                solver_success=bool(result.success),
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
            self.last_projection_candidates.append(candidate)
            if accepted:
                if self.last_pose_projection_reason is None:
                    self.last_pose_projection_reason = reason
                projected.append((score, solution_rad, candidate))
        if not projected:
            return None
        projected.sort(key=lambda item: item[0])
        return projected[0]



class PiperDifferentialIK:
    """IsaacLab-style damped least-squares differential IK for Piper.

    This solver does not search for a complete IK branch. It computes a local
    joint update around the measured current joints:

        dq = J.T @ inv(J @ J.T + lambda^2 I) @ pose_error
        q_cmd = q_current + dq

    The Jacobian is estimated from Piper SDK forward kinematics by central
    finite differences, so it can run on the real robot host without Isaac.
    """

    def __init__(
        self,
        fk,
        max_joint_step_deg: float,
        min_joint_limit_margin_deg: float = 0.0,
        lambda_val: float = 0.01,
        finite_difference_eps_rad: float = 1e-4,
        position_gain: float = 1.0,
        rotation_gain: float = 1.0,
    ) -> None:
        self.fk = fk
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.min_joint_limit_margin_deg = float(min_joint_limit_margin_deg)
        self.lambda_val = float(lambda_val)
        self.finite_difference_eps_rad = float(finite_difference_eps_rad)
        self.position_gain = float(position_gain)
        self.rotation_gain = float(rotation_gain)
        self.lower_deg = JOINT_LOWER_DEG + self.min_joint_limit_margin_deg
        self.upper_deg = JOINT_UPPER_DEG - self.min_joint_limit_margin_deg
        self.lower_rad = np.radians(self.lower_deg)
        self.upper_rad = np.radians(self.upper_deg)

    def _fk_pose(self, joints_rad: np.ndarray) -> tuple[np.ndarray, Rotation]:
        pose = np.asarray(self.fk.CalFK(joints_rad.tolist())[-1], dtype=np.float64)
        return pose[:3] / 1000.0, Rotation.from_euler("xyz", np.radians(pose[3:]))

    @staticmethod
    def _pose_error(
        current_position_m: np.ndarray,
        current_rotation: Rotation,
        target_position_m: np.ndarray,
        target_rotation: Rotation,
    ) -> tuple[np.ndarray, np.ndarray]:
        position_error = target_position_m - current_position_m
        rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()
        return position_error, rotation_error

    def _numerical_jacobian(self, current_joint_rad: np.ndarray) -> np.ndarray:
        eps = self.finite_difference_eps_rad
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for joint_index in range(6):
            plus = current_joint_rad.copy()
            minus = current_joint_rad.copy()
            plus[joint_index] += eps
            minus[joint_index] -= eps
            plus = np.clip(plus, self.lower_rad, self.upper_rad)
            minus = np.clip(minus, self.lower_rad, self.upper_rad)
            actual_delta = plus[joint_index] - minus[joint_index]
            if actual_delta <= 1e-12:
                continue
            plus_position, plus_rotation = self._fk_pose(plus)
            minus_position, minus_rotation = self._fk_pose(minus)
            jacobian[:3, joint_index] = (plus_position - minus_position) / actual_delta
            jacobian[3:, joint_index] = (plus_rotation * minus_rotation.inv()).as_rotvec() / actual_delta
        return jacobian

    def solve(
        self,
        target_position_m: np.ndarray,
        target_euler_xyz_rad: np.ndarray,
        current_joint_rad: np.ndarray,
    ) -> dict[str, object]:
        started = time.monotonic()
        target_position_m = np.asarray(target_position_m, dtype=np.float64)
        target_euler_xyz_rad = np.asarray(target_euler_xyz_rad, dtype=np.float64)
        current_joint_rad = np.asarray(current_joint_rad, dtype=np.float64)
        if target_position_m.shape != (3,) or target_euler_xyz_rad.shape != (3,):
            raise ValueError("Differential IK target must contain 3D position and Euler rotation")
        if current_joint_rad.shape != (6,):
            raise ValueError("Current Piper joints must contain six angles")
        if not (
            np.isfinite(target_position_m).all()
            and np.isfinite(target_euler_xyz_rad).all()
            and np.isfinite(current_joint_rad).all()
        ):
            raise ValueError("Differential IK input contains NaN or Inf")

        current_joint_rad = np.clip(current_joint_rad, self.lower_rad, self.upper_rad)
        current_deg = np.degrees(current_joint_rad)
        target_rotation = Rotation.from_euler("xyz", target_euler_xyz_rad)
        current_position, current_rotation = self._fk_pose(current_joint_rad)
        position_error, rotation_error = self._pose_error(
            current_position, current_rotation, target_position_m, target_rotation
        )
        pose_error = np.concatenate(
            (self.position_gain * position_error, self.rotation_gain * rotation_error)
        )
        jacobian = self._numerical_jacobian(current_joint_rad)
        lambda_matrix = (self.lambda_val**2) * np.eye(6, dtype=np.float64)
        try:
            delta_joint_rad = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + lambda_matrix, pose_error
            )
        except np.linalg.LinAlgError as error:
            diagnostics = {
                "solver": "numerical_dls_differential_ik",
                "target_sdk_link6": {
                    "position_m": target_position_m,
                    "euler_xyz_rad": target_euler_xyz_rad,
                    "euler_xyz_deg": np.degrees(target_euler_xyz_rad),
                },
                "current_joint_degrees": current_deg,
                "current_position_m": current_position,
                "current_euler_xyz_rad": current_rotation.as_euler("xyz"),
                "position_error_m": position_error,
                "rotation_error_axis_angle_rad": rotation_error,
                "jacobian": jacobian,
                "lambda_val": self.lambda_val,
                "solve_seconds": time.monotonic() - started,
                "error": str(error),
            }
            raise HostIKError("Differential IK linear solve failed", diagnostics) from error

        requested_max_step_deg = float(np.max(np.abs(np.degrees(delta_joint_rad))))
        if requested_max_step_deg > self.max_joint_step_deg:
            joint_step_scale = self.max_joint_step_deg / requested_max_step_deg
        else:
            joint_step_scale = 1.0
        command_rad = current_joint_rad + joint_step_scale * delta_joint_rad
        unclipped_command_rad = command_rad.copy()
        command_rad = np.clip(command_rad, self.lower_rad, self.upper_rad)
        command_deg = np.degrees(command_rad)
        command_position, command_rotation = self._fk_pose(command_rad)
        command_position_error, command_rotation_error = self._pose_error(
            command_position, command_rotation, target_position_m, target_rotation
        )
        command_limit_margin_deg = float(
            np.min(np.minimum(command_deg - JOINT_LOWER_DEG, JOINT_UPPER_DEG - command_deg))
        )
        clipped_by_limits = bool(np.max(np.abs(command_rad - unclipped_command_rad)) > 1e-12)
        return {
            "solver": "numerical_dls_differential_ik",
            "target_sdk_link6": {
                "position_m": target_position_m,
                "euler_xyz_rad": target_euler_xyz_rad,
                "euler_xyz_deg": np.degrees(target_euler_xyz_rad),
            },
            "current_joint_degrees": current_deg,
            "joint_lower_degrees": JOINT_LOWER_DEG,
            "joint_upper_degrees": JOINT_UPPER_DEG,
            "current_position_m": current_position,
            "current_euler_xyz_rad": current_rotation.as_euler("xyz"),
            "position_error_m": position_error,
            "rotation_error_axis_angle_rad": rotation_error,
            "pose_error": pose_error,
            "jacobian": jacobian,
            "lambda_val": self.lambda_val,
            "finite_difference_eps_rad": self.finite_difference_eps_rad,
            "position_gain": self.position_gain,
            "rotation_gain": self.rotation_gain,
            "delta_joint_degrees": np.degrees(delta_joint_rad),
            "selected_joint_degrees": command_deg,
            "selected_joint_radians": command_rad,
            "ik_solution_joint_degrees": command_deg,
            "ik_solution_joint_radians": command_rad,
            "selected_score": float(np.linalg.norm(pose_error)),
            "selected": {
                "joint_degrees": command_deg,
                "position_error_m": float(np.linalg.norm(command_position_error)),
                "rotation_error_rad": float(np.linalg.norm(command_rotation_error)),
                "minimum_limit_margin_deg": command_limit_margin_deg,
                "maximum_step_from_current_deg": float(np.max(np.abs(command_deg - current_deg))),
                "accepted": True,
                "rejection_reason": None,
            },
            "exact_solution_available": True,
            "pose_projected": False,
            "pose_projection_reason": None,
            "joint_step_limited": joint_step_scale < 1.0,
            "joint_step_scale": float(joint_step_scale),
            "requested_max_joint_step_deg": requested_max_step_deg,
            "commanded_max_joint_step_deg": float(np.max(np.abs(command_deg - current_deg))),
            "commanded_minimum_limit_margin_deg": command_limit_margin_deg,
            "commanded_position_error_m": float(np.linalg.norm(command_position_error)),
            "commanded_rotation_error_rad": float(np.linalg.norm(command_rotation_error)),
            "joint_limit_clipped": clipped_by_limits,
            "solve_seconds": time.monotonic() - started,
        }
