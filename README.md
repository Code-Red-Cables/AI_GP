# Keyboard teleop

Minimal MAVLink client: fly the sim with your keyboard. No YOLO, vision, or VIO.

## Run (Windows / Parallels)

```powershell
cd C:\Mac\Home\Documents\AI_GP
pip install -r requirements.txt
python main.py
```

Click the terminal so it has keyboard focus, then:

| Key | Action |
|-----|--------|
| W / S | forward / back |
| A / D | strafe left / right |
| Space / C | climb / descend |
| Q / E | yaw left / right |
| X | hover (clear sticks) |
| Ctrl+C | disarm and quit |

Optional env vars: `TELEOP_SPEED`, `TELEOP_VSPEED`, `TELEOP_YAW_RATE_DPS`, `RESET_SIM_ON_START=1`.
