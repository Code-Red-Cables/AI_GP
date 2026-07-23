# Control & Reset Probe Report

## Reset (cmd 31000)

- active_gate: before=0 -> after=0 (check)
- race_start_ms: before=3299 -> after=3298 (changed)
- time to first fresh frame after reset: 26 ms

## Control response (raw physical commands)

baseline gyro (rad/s): x=-0.000 y=+0.000 z=+0.000

| pulse | cmd | measured axis (mean, baseline-subtracted) | inferred sign |
|---|---|---|---|
| roll_pos (+1.5) | +rate | Δgyro_x=-4.001 | -1 |
| pitch_pos (+1.5) | +rate | Δgyro_y=-3.477 | -1 |
| yaw_pos (+1.5) | +rate | Δgyro_z=-0.788 | -1 |
| thrust_up (+0.3) | +thrust | Δacc_z=-16.539 | responds |

## Recommended config.action signs
For each axis: if commanding +rate produced +gyro, keep sign +1; if it
produced -gyro, set that rate_sign to -1 (the sim inverts that axis).

- rate_sign_roll:  -1
- rate_sign_pitch: -1   (config default is -1)
- rate_sign_yaw:   -1
