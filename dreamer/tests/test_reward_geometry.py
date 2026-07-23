"""Reward geometry unit tests (no simulator needed). Verifies each component points the
right direction and that anti-exploit clipping / one-shot logic hold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dreamer_drone.config import RewardConfig
from dreamer_drone.env.reward import RewardComputer, StepContext


def _fresh(cfg=None):
    rc = RewardComputer(cfg or RewardConfig())
    rc.reset()
    return rc


def test_gate_pass_positive_and_backtrack_negative():
    rc = _fresh()
    rc.compute(StepContext(active_gate=0))            # establish baseline
    out = rc.compute(StepContext(active_gate=1))      # crossed a gate
    assert out.gate_pass == RewardConfig().w_gate
    back = rc.compute(StepContext(active_gate=0))      # went backwards
    assert back.backtrack < 0


def test_finish_rewards_once():
    rc = _fresh()
    a = rc.compute(StepContext(finished=True))
    b = rc.compute(StepContext(finished=True))
    assert a.finish == RewardConfig().w_finish
    assert b.finish == 0.0


def test_collision_penalizes():
    rc = _fresh()
    out = rc.compute(StepContext(collision_threat=2))
    assert out.collision == -RewardConfig().w_collision


def test_time_penalty_scales_with_sim_time():
    rc = _fresh()
    out = rc.compute(StepContext(dt_sim=0.5))
    assert abs(out.time_penalty - (-RewardConfig().w_time * 0.5)) < 1e-9


def test_vision_progress_rewards_approaching_gate():
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))   # baseline
    grow = rc.compute(StepContext(gate_visible=True, gate_area_px=40000.0))  # bigger => closer
    assert grow.progress > 0
    shrink = rc.compute(StepContext(gate_visible=True, gate_area_px=1000.0))  # smaller => farther
    assert shrink.progress < 0


def test_progress_is_clipped():
    cfg = RewardConfig(progress_clip=0.5)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=1.0))
    out = rc.compute(StepContext(gate_visible=True, gate_area_px=1e9))
    assert out.progress <= cfg.w_progress * cfg.progress_clip + 1e-6


def test_privileged_progress_used_when_enabled():
    cfg = RewardConfig(use_privileged_progress=True)
    rc = _fresh(cfg)
    rc.compute(StepContext(dist_to_gate=10.0))          # baseline
    out = rc.compute(StepContext(dist_to_gate=9.0))     # closed 1 m
    assert out.progress > 0
    assert out.raw["progress_privileged"] == 1.0


def test_offcourse_when_no_gate_visible():
    rc = _fresh()
    out = rc.compute(StepContext(gate_visible=False))
    assert out.offcourse == -RewardConfig().w_offcourse


def test_control_cost_penalizes_action_change():
    rc = _fresh()
    out = rc.compute(StepContext(action=[1.0, 1.0, 1.0, 1.0], prev_action=[0.0, 0.0, 0.0, 0.0]))
    assert out.control < 0


def test_total_is_sum_of_components():
    rc = _fresh()
    rc.compute(StepContext(active_gate=0))
    out = rc.compute(StepContext(active_gate=1, dt_sim=0.1, collision_threat=0,
                                 gate_visible=True, gate_area_px=100.0,
                                 action=[0, 0, 0, 0], prev_action=[0, 0, 0, 0]))
    expected = (out.progress + out.gate_pass + out.finish + out.time_penalty
                + out.collision + out.control + out.offcourse + out.backtrack)
    assert abs(out.total - expected) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} reward-geometry tests passed.")
