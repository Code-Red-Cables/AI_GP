# Simulator GUI Automation Ideas

The DCL Flight Simulator requires several manual steps to reach a state where it streams telemetry:
1.  **Login:** Enter credentials and click "Login".
2.  **Menu Navigation:** Select "Virtual Qualifier".
3.  **Race Selection:** Choose "Round 1".
4.  **Start:** Click "Start Race".

### Potential Automation Strategies

#### 1. PyAutoGUI (Image Recognition)
- Use `pyautogui.locateOnScreen()` to find buttons by their icons.
- Use `pyautogui.click()` to perform the interactions.
- **Pros:** Cross-platform, relatively simple to script.
- **Cons:** Sensitive to screen resolution, window position, and UI theme changes.

#### 2. WinAppDriver / Appium (Windows UI Automation)
- Use Microsoft's WinAppDriver to inspect the application's UI tree.
- Identify elements by their `AutomationId` or `Name`.
- **Pros:** More robust than image recognition; works even if the window is partially obscured.
- **Cons:** Requires installing the WinAppDriver service; simulator elements might be "flat" (custom-drawn) and not visible to UI Automation.

#### 3. Command Line Arguments
- Check if `FlightSim.exe` or `DCGame-Win64-Shipping.exe` supports any hidden CLI flags to skip the menu (e.g., `-race=VQ1`, `-autologin`).
- **Pros:** Most robust method.
- **Cons:** Likely undocumented and might not exist.

### Future Implementation Note
If future agents need to automate this, they should start by capturing screenshots of the login and menu screens to build a library of "targets" for a tool like `PyAutoGUI`.
