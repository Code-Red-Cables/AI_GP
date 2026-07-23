"""Privileged-information leakage audit (Phase 10).

Asserts, without a simulator, that:
  1. the LEGAL observation dict contains only whitelisted keys/fields;
  2. importing the deployment controller pulls in NO privileged module;
  3. the deployed obs vector schema is exactly the audited LEGAL schema.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dreamer_drone.config import ObsConfig
from dreamer_drone.env import observation_builder as ob
from dreamer_drone.env.spaces import VECTOR_DIM, VECTOR_OBS_FIELDS

# Modules that must NEVER be reachable from the deployed policy.
PRIVILEGED_MODULES = {
    "dreamer_drone.sim.privileged_state",
    "dreamer_drone.env.reward",
    "dreamer_drone.env.termination",
    "dreamer_drone.env.curriculum",
}

LEGAL_OBS_KEYS = {"image", "vector", "valid"}


def test_built_obs_has_only_legal_keys():
    obs = ob.build_obs(
        frame_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
        imu={"xgyro": 0.1, "ygyro": 0.0, "zgyro": 0.0,
             "xacc": 0.0, "yacc": 0.0, "zacc": 9.8},
        prev_action_norm=np.zeros(4, dtype=np.float32),
        dt=0.033, cfg=ObsConfig(),
    )
    assert set(obs.keys()) == LEGAL_OBS_KEYS
    assert obs["vector"].shape[0] == VECTOR_DIM
    # no privileged field names leaked into the vector schema
    banned = {"gate_x", "gate_y", "gate_z", "pos_x", "pos_y", "pos_z",
              "active_gate", "collision", "dist_to_gate", "race_finish", "yaw"}
    assert banned.isdisjoint(set(VECTOR_OBS_FIELDS))


def test_deploy_controller_imports_no_privileged_module():
    # ensure a clean import graph measurement
    for m in list(sys.modules):
        if m.startswith("dreamer_drone"):
            del sys.modules[m]
    importlib.import_module("dreamer_drone.deploy.controller")
    leaked = PRIVILEGED_MODULES & set(sys.modules)
    assert not leaked, f"deploy.controller leaked privileged modules: {leaked}"


def test_observation_builder_imports_no_privileged_module():
    for m in list(sys.modules):
        if m.startswith("dreamer_drone"):
            del sys.modules[m]
    importlib.import_module("dreamer_drone.env.observation_builder")
    leaked = PRIVILEGED_MODULES & set(sys.modules)
    assert not leaked, f"observation_builder leaked privileged modules: {leaked}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} leakage-audit tests passed.")
