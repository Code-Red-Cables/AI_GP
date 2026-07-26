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


def test_gate_strike_charged_softer_than_environment():
    """A frame clip is a near-miss of a threading attempt, not a crash to fear."""
    cfg = RewardConfig(w_collision=8.0, w_collision_gate=3.0)
    rc = _fresh(cfg)
    env_hit = rc.compute(StepContext(collision_threat=1, collision_is_gate=False))
    gate_hit = rc.compute(StepContext(collision_threat=1, collision_is_gate=True))
    assert env_hit.collision == -8.0
    assert gate_hit.collision == -3.0


def test_time_penalty_scales_with_sim_time():
    rc = _fresh()
    out = rc.compute(StepContext(dt_sim=0.5))
    assert abs(out.time_penalty - (-RewardConfig().w_time * 0.5)) < 1e-9


def test_vision_progress_rewards_approaching_gate():
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))   # baseline
    grow = rc.compute(StepContext(gate_visible=True, gate_area_px=12000.0))  # bigger => closer
    assert grow.progress > 0
    # high-water credit: retreating pays nothing (not negative), and re-approaching
    # previously-achieved size pays nothing either — each px of apparent size pays once
    shrink = rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))
    assert shrink.progress == 0.0
    regrow = rc.compute(StepContext(gate_visible=True, gate_area_px=12000.0))
    assert regrow.progress == 0.0
    new_high = rc.compute(StepContext(gate_visible=True, gate_area_px=13000.0))
    assert new_high.progress > 0


def test_vision_progress_wobble_farm_pays_zero():
    """Regression for the dropout ratchet: grow -> dropout -> re-acquire smaller ->
    regrow must NOT pay again (measured +13/episode farmed this way with 0 gates)."""
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))
    rc.compute(StepContext(gate_visible=True, gate_area_px=12000.0))  # pays once
    total = 0.0
    for _ in range(50):  # wobble: lose detection, re-acquire small, grow back
        rc.compute(StepContext(gate_visible=False))
        rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))
        total += rc.compute(StepContext(gate_visible=True, gate_area_px=11000.0)).progress
        total += rc.compute(StepContext(gate_visible=True, gate_area_px=12000.0)).progress
    assert total == 0.0


def test_vision_progress_resets_after_gate_pass():
    rc = _fresh()
    rc.compute(StepContext(active_gate=0, gate_visible=True, gate_area_px=40000.0))
    rc.compute(StepContext(active_gate=1, gate_visible=True, gate_area_px=40000.0))  # passed
    # chasing the NEXT gate: fresh segment, growth pays even though 200px was achieved
    rc.compute(StepContext(active_gate=1, gate_visible=True, gate_area_px=10000.0))
    out = rc.compute(StepContext(active_gate=1, gate_visible=True, gate_area_px=12000.0))
    assert out.progress > 0


def test_progress_is_clipped():
    # disable the target-switch jump check so the clip path itself is exercised
    cfg = RewardConfig(progress_clip=0.5, progress_area_jump=0.0)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=1.0))
    out = rc.compute(StepContext(gate_visible=True, gate_area_px=1e9))
    assert out.progress <= cfg.w_progress * cfg.progress_clip + 1e-6


def test_progress_target_switch_rebaselines():
    """A large one-step area jump = the detector switched objects: no spurious reward."""
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))       # baseline
    switch = rc.compute(StepContext(gate_visible=True, gate_area_px=40000.0))  # 4x area jump
    assert switch.progress == 0.0
    # ... but the new target becomes the baseline and smooth growth counts again
    grow = rc.compute(StepContext(gate_visible=True, gate_area_px=44000.0))
    assert grow.progress > 0


def test_progress_center_jump_rebaselines():
    """A large one-step center jump = target switch even if the areas happen to match."""
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0, gate_center=(0.2, 0.5)))
    out = rc.compute(StepContext(gate_visible=True, gate_area_px=11000.0, gate_center=(0.9, 0.5)))
    assert out.progress == 0.0


def test_progress_dropout_resets_baseline():
    """Losing the detection clears the baseline: re-acquisition yields no stale delta."""
    rc = _fresh()
    rc.compute(StepContext(gate_visible=True, gate_area_px=10000.0))
    rc.compute(StepContext(gate_visible=False))
    reacq = rc.compute(StepContext(gate_visible=True, gate_area_px=11000.0))
    assert reacq.progress == 0.0


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


def test_centering_pays_recentering_and_charges_decentering():
    cfg = RewardConfig(w_center=2.0)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=100.0, gate_center=(0.3, 0.5)))
    toward = rc.compute(StepContext(gate_visible=True, gate_area_px=100.0, gate_center=(0.4, 0.5)))
    assert abs(toward.centering - 2.0 * 0.1) < 1e-9
    away = rc.compute(StepContext(gate_visible=True, gate_area_px=100.0, gate_center=(0.1, 0.5)))
    assert abs(away.centering - (-2.0 * 0.3)) < 1e-9


def test_centering_charges_losing_the_gate():
    """Drifting the gate off-frame must be charged (worst potential), not escape free."""
    cfg = RewardConfig(w_center=2.0)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=100.0, gate_center=(0.5, 0.5)))
    lost = rc.compute(StepContext(gate_visible=False))
    assert abs(lost.centering - (-2.0 * 0.5)) < 1e-9


def test_centering_loop_nets_zero():
    """Pure potential shaping: any centered->lost->reacquired->centered cycle sums to 0,
    so it cannot be hover-farmed (the w_offcourse / dropout-ratchet failure modes)."""
    cfg = RewardConfig(w_center=2.0)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=100.0, gate_center=(0.5, 0.5)))
    total = 0.0
    for _ in range(20):
        total += rc.compute(StepContext(gate_visible=False)).centering
        total += rc.compute(StepContext(gate_visible=True, gate_area_px=100.0,
                                        gate_center=(0.2, 0.5))).centering
        total += rc.compute(StepContext(gate_visible=True, gate_area_px=100.0,
                                        gate_center=(0.5, 0.5))).centering
    assert abs(total) < 1e-9


def test_centering_not_charged_across_gate_pass():
    """Passing a gate switches view to the NEXT gate; that jump must not be charged."""
    cfg = RewardConfig(w_center=2.0)
    rc = _fresh(cfg)
    rc.compute(StepContext(active_gate=0, gate_visible=True, gate_area_px=100.0,
                           gate_center=(0.5, 0.5)))
    passed = rc.compute(StepContext(active_gate=1, gate_visible=False))
    # gate_pass fired this step; the centering baseline resets instead of charging -0.5
    assert passed.gate_pass > 0
    assert passed.centering == 0.0
    # baseline re-armed at "not visible": turning to FIND the next gate pays positively
    found = rc.compute(StepContext(active_gate=1, gate_visible=True, gate_area_px=50.0,
                                   gate_center=(0.3, 0.5)))
    assert abs(found.centering - 2.0 * (0.5 - 0.2)) < 1e-9


def test_centering_close_range_sight_loss_is_held_not_charged():
    """Threading takes the gate out of frame: at close range (large last-seen area),
    sight loss holds φ neutral so the pass can register instead of being taxed."""
    cfg = RewardConfig(w_center=4.0, center_hold_sqrt_px=90.0, center_hold_steps=8)
    rc = _fresh(cfg)
    rc.compute(StepContext(active_gate=0, gate_visible=True, gate_area_px=10000.0,  # sqrt=100
                           gate_center=(0.5, 0.5)))
    blind = rc.compute(StepContext(active_gate=0, gate_visible=False))
    assert blind.centering == 0.0          # held, not charged -2
    passed = rc.compute(StepContext(active_gate=1, gate_visible=False))
    assert passed.gate_pass > 0
    assert passed.centering == 0.0


def test_centering_distant_sight_loss_still_charged():
    cfg = RewardConfig(w_center=4.0, center_hold_sqrt_px=90.0, center_hold_steps=8)
    rc = _fresh(cfg)
    rc.compute(StepContext(gate_visible=True, gate_area_px=400.0,   # sqrt=20: far away
                           gate_center=(0.5, 0.5)))
    lost = rc.compute(StepContext(gate_visible=False))
    assert abs(lost.centering - (-4.0 * 0.5)) < 1e-9


def test_centering_hold_expires_without_a_pass():
    """Climbing over the gate (blind, no pass) gets charged once the hold runs out."""
    cfg = RewardConfig(w_center=4.0, center_hold_sqrt_px=90.0, center_hold_steps=3)
    rc = _fresh(cfg)
    rc.compute(StepContext(active_gate=0, gate_visible=True, gate_area_px=10000.0,
                           gate_center=(0.5, 0.5)))
    total = 0.0
    for _ in range(5):
        total += rc.compute(StepContext(active_gate=0, gate_visible=False)).centering
    assert abs(total - (-4.0 * 0.5)) < 1e-9   # charged exactly once, after the window


def test_total_is_sum_of_components():
    rc = _fresh()
    rc.compute(StepContext(active_gate=0))
    out = rc.compute(StepContext(active_gate=1, dt_sim=0.1, collision_threat=0,
                                 gate_visible=True, gate_area_px=100.0,
                                 action=[0, 0, 0, 0], prev_action=[0, 0, 0, 0]))
    expected = (out.progress + out.gate_pass + out.finish + out.time_penalty
                + out.collision + out.control + out.offcourse + out.backtrack
                + out.alive + out.centering)
    assert abs(out.total - expected) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} reward-geometry tests passed.")
