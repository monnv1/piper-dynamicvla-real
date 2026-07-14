from __future__ import annotations

import numpy as np


# Calibrated physical standby/start configuration. Keep all return tools and
# the deployment runtime on this single definition.
TRAINING_START_DEG = np.asarray(
    [0.0, 70.913, -60.913, 0.0, 58.398, 0.0],
    dtype=np.float64,
)
TRAINING_START_RAD = np.radians(TRAINING_START_DEG)
