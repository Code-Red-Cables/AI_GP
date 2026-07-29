import time

import config

SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550


def _z_sources():
    """Altitude keys for crash logic, best first.

    VQ2 publishes no LOCAL_POSITION_NED, so this is always ('position_ned',)
    there. The VQ1 build does publish it, which makes crash detection more
    reliable but no longer representative — set CRASH_USE_SIM_ODOMETRY=0 to
    rehearse VQ2 behaviour on the richer build.
    """
    if config.CRASH_USE_SIM_ODOMETRY:
        return ('local_position_ned', 'position_ned')
    return ('position_ned',)


def _read_z(shared_data):
    """Prefer sim odometry for crash logic; EKF drifts across resets."""
    for key in _z_sources():
        pos = shared_data.get(key) or {}
        z = pos.get('z')
        if z is None:
            continue
        try:
            return float(z)
        except (TypeError, ValueError):
            continue
    return None


def _max_abs_z(shared_data):
    """Worst |z| across sources — catch EKF runaway even if local looks fine."""
    best = None
    for key in _z_sources():
        pos = shared_data.get(key) or {}
        z = pos.get('z')
        if z is None:
            continue
        try:
            az = abs(float(z))
        except (TypeError, ValueError):
            continue
        if best is None or az > best:
            best = az
    return best


class CrashMonitor:
    """Detect floor crashes relative to the post-reset spawn, not absolute NED."""

    def __init__(self):
        self.z0 = None
        self.peak_climb = 0.0
        self.crash_since = None
        self.gate_slam_until = 0.0
        self.last_reset_at = 0.0
        self.attempt = 0
        self.grace_until = 0.0

    def note_armed(self, shared_data):
        self.attempt += 1
        z = _read_z(shared_data)
        self.z0 = z if z is not None else 0.0
        self.peak_climb = 0.0
        self.crash_since = None
        self.gate_slam_until = 0.0
        self.grace_until = time.monotonic() + max(
            2.0, config.SIM_RESET_SETTLE_S + 0.5
        )
        print(
            f'[SIM] attempt={self.attempt} z0={self.z0:.3f}',
            flush=True,
        )

    def update(self, shared_data, now: float) -> bool:
        """Return True once a floor crash is confirmed."""
        if now < self.grace_until:
            return False
        if now - self.last_reset_at < config.CRASH_RESET_COOLDOWN_S:
            return False

        # 071313: auto-reset fired 30 ms before GATE_PASSED while phase=coast.
        # Do not yank the drone mid punch-through.
        path = shared_data.get('kalman_path') or {}
        phase = str(path.get('phase') or '')
        col = shared_data.get('collision') or {}
        if col.get('id') == 1001:  # Gate
            try:
                impulse = float(col.get('impulse') or 0.0)
                ts_ns = int(col.get('ts') or 0)
            except (TypeError, ValueError):
                impulse, ts_ns = 0.0, 0
            age_s = (time.time_ns() - ts_ns) * 1e-9 if ts_ns else 999.0
            # Latch: follow-up weak Gate hits were clearing crash_since (0752).
            if impulse >= 1.5 and 0.0 <= age_s <= 0.6:
                self.gate_slam_until = max(self.gate_slam_until, now + 1.2)
        gate_slam = now < self.gate_slam_until

        # 0759: through with area~116k was reset by Gate contact before the
        # sim could score. Ignore slams during punch; reset only after coast.
        # Still abort on EKF/world runaway even mid-punch (080022 freefall).
        z_run_early = _max_abs_z(shared_data)
        if (
            z_run_early is not None
            and z_run_early > 15.0
            and self.peak_climb >= 0.30
        ):
            if self.crash_since is None:
                self.crash_since = now
            return (now - self.crash_since) >= config.CRASH_CONFIRM_S
        # Hover clears phase to 'hover' — do not leave stale coast masking
        # freefall resets (0825).
        if phase in ('through', 'coast'):
            self.crash_since = None
            return False
        # 0910/0926: Gate slam during post-pass seek/coast/commit is often
        # NEXT-gate contact — don't abort before score. Do NOT protect
        # approach (0928: first-gate scrapes never reset → 0 GATE_PASSED).
        if gate_slam and phase in ('commit', 'seek', 'coast'):
            # Require an active post-pass hunt flag so a stale seek phase
            # cannot mask first-gate scrapes.
            if shared_data.get('post_pass_hunt'):
                self.crash_since = None
                return False

        z = _read_z(shared_data)
        if z is None:
            return False
        if self.z0 is None:
            self.z0 = z

        climbed = self.z0 - z  # positive when above spawn
        if climbed > self.peak_climb:
            self.peak_climb = climbed

        # EKF/odom runaway (071945/0748): local can look fine while EKF blows.
        # 0829: after a bad reset, peak_climb stayed ~0 while |z|→20 and the
        # drone hovered in freefall until the time bound — don't require climb.
        z_run = _max_abs_z(shared_data)
        if z_run is not None and z_run > 15.0:
            if self.crash_since is None:
                self.crash_since = now
            return (now - self.crash_since) >= config.CRASH_CONFIRM_S
        if abs(z - (self.z0 or 0.0)) > 25.0:
            if self.crash_since is None:
                self.crash_since = now
            return (now - self.crash_since) >= config.CRASH_CONFIRM_S

        # Only treat "below spawn" as a crash after we actually flew —
        # otherwise a bad post-reset z0 causes an immediate re-trigger.
        fallen_below_spawn = (
            self.peak_climb >= 0.40
            and climbed <= -config.CRASH_FLOOR_D_M
        )
        # Was airborne then lost most of the climb (tumble / drop).
        dropped_from_peak = (
            self.peak_climb >= 0.50
            and climbed <= self.peak_climb - 0.80
            and climbed < 0.05
        )

        env_slam = False
        col = shared_data.get('collision') or {}
        if col.get('id') == 1002:
            try:
                impulse = float(col.get('impulse') or 0.0)
                ts_ns = int(col.get('ts') or 0)
            except (TypeError, ValueError):
                impulse, ts_ns = 0.0, 0
            age_s = (time.time_ns() - ts_ns) * 1e-9 if ts_ns else 999.0
            if (
                impulse >= config.CRASH_ENV_IMPULSE_MIN
                and 0.0 <= age_s <= 0.5
                and self.peak_climb >= 0.35
                and climbed < 0.20
            ):
                env_slam = True

        crashed = fallen_below_spawn or dropped_from_peak or env_slam
        if crashed:
            if self.crash_since is None:
                self.crash_since = now
            return (now - self.crash_since) >= config.CRASH_CONFIRM_S

        self.crash_since = None
        return False


def _log_attempt(shared_data, monitor: CrashMonitor, logger) -> None:
    """Snapshot one-line attempt summary for the learn/fix loop."""
    dual = shared_data.get('dual_gate_pnp') or {}
    det = shared_data.get('gate_detection') or {}
    path = shared_data.get('kalman_path') or {}
    msg = (
        f'attempt={monitor.attempt} peak_climb={monitor.peak_climb:.2f} '
        f'range={dual.get("gate1_range_m")} area={det.get("area_px")} '
        f'phase={path.get("phase")} source={path.get("source")}'
    )
    print(f'[LEARN] {msg}', flush=True)
    if logger is not None:
        logger.log_event('ATTEMPT_END', msg)


def _early_start_hold() -> None:
    """Wait before arm so the OLD sim does not early-start DQ."""
    hold_s = max(0.0, float(config.EARLY_START_HOLD_S))
    if hold_s <= 0.0:
        return
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)


def _wait_pad_vision(shared_data, timeout_s: float = 45.0) -> bool:
    """Block until vision sees gate 1 (pad facing the course).

    Assist mode accepts a YOLO box; kalman mode wants dual-gate PnP.
    """
    want_pnp = config.FLIGHT_MODE != 'assist'
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    need = 'DUAL_PNP (gate1_body)' if want_pnp else 'YOLO gate or DUAL_PNP'
    print(
        f'[PAD] waiting for {need} before arm — '
        'start the race on the pad facing gate 1 if this hangs...',
        flush=True,
    )
    while time.monotonic() < deadline:
        with shared_data['lock']:
            dual = shared_data.get('dual_gate_pnp') or {}
            det = shared_data.get('gate_detection') or {}
            vstate = str(shared_data.get('vision_state') or '')
        n_solved = int(dual.get('n_solved') or 0)
        has_body = dual.get('gate1_body') is not None
        area = det.get('area_px') if isinstance(det, dict) else None
        try:
            area_ok = area is not None and float(area) >= 800.0
        except (TypeError, ValueError):
            area_ok = False
        if has_body and (n_solved >= 1 or vstate == 'DUAL_PNP'):
            rng = dual.get('gate1_range_m')
            print(
                f'[PAD] ready vision_state={vstate or "DUAL_PNP"} '
                f'n_solved={n_solved} range={rng}',
                flush=True,
            )
            return True
        if not want_pnp and area_ok:
            print(
                f'[PAD] ready YOLO area={float(area):.0f} '
                f'vision_state={vstate or "?"}',
                flush=True,
            )
            return True
        time.sleep(0.2)
    print(
        f'[PAD] FAIL — no gate after {timeout_s:.0f}s '
        '(not on pad / race not started)',
        flush=True,
    )
    return False


def _reset_after_crash(
    controller, planner, shared_data, logger, state_estimator, monitor
) -> None:
    _log_attempt(shared_data, monitor, logger)
    print('[SIM] floor crash — resetting drone (cmd 31000)', flush=True)
    if logger is not None:
        logger.log_event('SIM_RESET', 'auto_floor_crash')
    shared_data['flight_started'] = False
    # 0859: clear vision before settle sleep so the RX thread drops stale
    # post-pass/association state while the sim resets.
    shared_data['vision_reset_episode'] = True
    shared_data['course_bearing'] = None
    shared_data['gate_detection'] = None
    shared_data['dual_gate_pnp'] = None
    shared_data['kalman_path'] = None
    shared_data['post_pass_hunt'] = False
    try:
        controller.disarm()
    except Exception:
        pass
    controller.send_sim_reset()
    time.sleep(max(0.0, config.SIM_RESET_SETTLE_S))
    resetter = getattr(planner, 'reset_episode', None)
    if callable(resetter):
        resetter()
    if state_estimator is not None:
        ekf_reset = getattr(state_estimator, 'reset_episode', None)
        if callable(ekf_reset):
            ekf_reset()
    shared_data['collision'] = None
    shared_data['planner_target'] = None
    # 0847: vision must drop sticky pre-pass bearings from the prior lap.
    shared_data['vision_reset_episode'] = True
    # 080022: after falling through the world, stale local Z stayed huge and
    # poisoned the next attempt. Clear until mavlink/EKF republish.
    shared_data['local_position_ned'] = None
    if not config.PERCEPTION_ONLY:
        _wait_pad_vision(shared_data)
        _early_start_hold()
        controller.arm()
        shared_data['flight_started'] = True
    monitor.last_reset_at = time.monotonic()
    monitor.note_armed(shared_data)
    print('[SIM] reset complete — re-armed', flush=True)


def run_racing():
    """Run the Q2 rate controller with the dual-gate PnP + EKF planner."""
    from setup import setup_components

    system_boot_ms = int(time.time() * 1000)
    shared_data = {}
    components = setup_components(
        shared_data,
        system_boot_ms,
        SIM_SERVER_UDP_IP,
        SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = components['planner']
    logger = components['logger']
    state_estimator = components.get('state_estimator')
    monitor = CrashMonitor()

    print(
        f'[MODE] flight={config.FLIGHT_MODE} planner={planner.name}',
        flush=True,
    )
    if config.FLIGHT_MODE == 'assist':
        print(
            '[MODE] assist = image nx/ny chase on manual attitude+hover '
            '(no EKF position steering)',
            flush=True,
        )
    if config.PERCEPTION_ONLY:
        print(
            '[SAFE] perception-only: not arming and not sending flight commands',
            flush=True,
        )
    else:
        _wait_pad_vision(shared_data)
        _early_start_hold()
        print('Arming drone...', flush=True)
        controller.arm()
        # Release the VIO's pre-flight ZUPT: dead-reckoning may only start
        # once the drone can actually leave its spawn point.
        shared_data['flight_started'] = True
        monitor.note_armed(shared_data)
    print('Control loop running -- Ctrl+C to exit', flush=True)

    run_started_at = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if (
                config.RUN_MAX_SECONDS > 0.0
                and now - run_started_at >= config.RUN_MAX_SECONDS
            ):
                print(
                    f'[SIM] bounded run reached '
                    f'{config.RUN_MAX_SECONDS:.1f}s',
                    flush=True,
                )
                break
            if (
                config.AUTO_RESET_ON_CRASH
                and not config.PERCEPTION_ONLY
                and monitor.update(shared_data, now)
            ):
                _reset_after_crash(
                    controller,
                    planner,
                    shared_data,
                    logger,
                    state_estimator,
                    monitor,
                )
                continue
            shared_data['planner_target'] = planner.compute_target(shared_data)
            if config.PERCEPTION_ONLY:
                time.sleep(1.0 / config.CONTROL_HZ)
            else:
                controller.update()
    except KeyboardInterrupt:
        pass
    finally:
        if not config.PERCEPTION_ONLY:
            controller.disarm()
        logger.stop()
        for name in ('ts_loop', 'mavlink_rx', 'vision_rx', 'state_estimator'):
            component = components.get(name)
            if component is not None:
                joiner = getattr(component, 'get_thread_for_join', None)
                if joiner is not None:
                    joiner().join(timeout=1.0)
        print('Client exited!', flush=True)


def main():
    run_racing()


if __name__ == '__main__':
    main()
