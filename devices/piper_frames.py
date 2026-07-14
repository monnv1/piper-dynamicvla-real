from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class PiperFrameTransform:
    """Convert poses between Piper SDK link6 and DynamicVLA's model TCP.

    ``sdk_to_model_*`` describes the fixed transform T_sdk_model. Poses use
    base_link as their common parent and static-axis XYZ Euler angles.
    """

    def __init__(
        self,
        sdk_to_model_translation_m: list[float],
        sdk_to_model_euler_xyz_rad: list[float],
    ) -> None:
        self.translation = self._vector(
            sdk_to_model_translation_m, "sdk_to_model_translation_m"
        )
        fixed_euler = self._vector(
            sdk_to_model_euler_xyz_rad, "sdk_to_model_euler_xyz_rad"
        )
        self.rotation = Rotation.from_euler("xyz", fixed_euler)

    @staticmethod
    def _vector(value: list[float] | np.ndarray, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{label} must contain three finite values")
        return vector

    @staticmethod
    def _model_euler(rotation: Rotation) -> np.ndarray:
        euler = rotation.as_euler("xyz")
        # DynamicVLA dataset preprocessing stores X and Z in [0, 2*pi).
        euler[[0, 2]] %= 2.0 * np.pi
        return euler

    def sdk_to_model_pose(
        self, position_m: np.ndarray, euler_xyz_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sdk_position = self._vector(position_m, "SDK position")
        sdk_rotation = Rotation.from_euler(
            "xyz", self._vector(euler_xyz_rad, "SDK Euler")
        )
        model_position = sdk_position + sdk_rotation.apply(self.translation)
        model_rotation = sdk_rotation * self.rotation
        return model_position, self._model_euler(model_rotation)

    def model_to_sdk_pose(
        self, position_m: np.ndarray, euler_xyz_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        model_position = self._vector(position_m, "model position")
        model_rotation = Rotation.from_euler(
            "xyz", self._vector(euler_xyz_rad, "model Euler")
        )
        sdk_rotation = model_rotation * self.rotation.inv()
        sdk_position = model_position - sdk_rotation.apply(self.translation)
        return sdk_position, sdk_rotation.as_euler("xyz")
