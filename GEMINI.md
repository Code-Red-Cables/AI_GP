# AI-GP Project Rules

- **DO NOT KILL THE SIMULATOR:** Never kill `DCGame-Win64-Shipping.exe` or `FlightSim.exe`. The simulator is a GUI application that requires the user to manually log in and start the race. If you kill it, you will break the testing loop and require manual user intervention to recover.
- **RESET SIMULATOR:** Use `powershell -ExecutionPolicy Bypass -File tools/reset_sim.ps1` to focus the sim and press **SPACE** automatically.
- **VIRTUAL ENV:** Always activate `myenv\Scripts\Activate.ps1` before running scripts.
- **TELEMETRY:** The simulator may send erratic `gate_idx` or position jumps. The planner has guards for this (strictly increasing index, world-jump rejection).

## Current Status (2026-06-13)
- **Vision:** Confident detections are refined via EMA and stored in `course_map.json`.
- **Navigation:** Drone now re-aims yaw toward `course_map` targets when vision is lost.
- **Persistence:** Course map auto-saves every 5s during recording.
- **Obstacle:** The drone currently passes Gate 0 but has been overshooting or missing Gate 1 due to altitude estimation noise or yaw lag.

## Tomorrow's Goals
1.  Complete a full "Learning Run" to populate `course_map.json` for all gates.
2.  Tune `KP_POS` and `MAX_SPEED` for smoother gate entry.
3.  Debug vertical altitude noise when approaching gates (possible pitch-coupling in vision).

