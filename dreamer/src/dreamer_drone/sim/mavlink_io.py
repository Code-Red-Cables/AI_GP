"""MAVLink transport: a background RX thread that parses telemetry into a thread-safe
latest-value store, plus TX helpers (control, arm, reset, timesync).

Adapted from the proven `../../mavlink_rx.py` + `../../timesync.py` + `../../controller.py`
send path. Struct layouts are preserved verbatim (they are hard-won). This layer stores
*all* messages; the LEGAL/PRIV split happens downstream (observation_builder vs
privileged_state) so this file stays a dumb, well-tested transport.
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Any, Optional

from pymavlink import mavutil

ENCAP_RACE_STATUS = 1
ENCAP_TRACK_INFO = 2

_ATTITUDE_IGNORE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


class MavlinkIO:
    def __init__(self, host: str, port: int, boot_ms: int, reset_cmd_id: int = 31000):
        self.conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
        self.boot_ms = boot_ms
        self.reset_cmd_id = reset_cmd_id
        self._lock = threading.Lock()
        self._store: dict[str, Any] = {}
        self._ts: dict[str, float] = {}          # wall-clock of last update per key
        self._track_chunks: dict[int, dict] = {}
        self._expected_chunks: dict[int, int] = {}
        self._collision_seq = 0                  # monotonically bumped on each collision
        self.running = False
        self.thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------
    def wait_heartbeat(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.conn.wait_heartbeat(timeout=1.0) is not None:
                return True
        return False

    def start(self) -> "MavlinkIO":
        self.running = True
        self.thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    # ---- store access ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def age(self, key: str) -> float:
        """Seconds since `key` last updated, or +inf if never."""
        with self._lock:
            t = self._ts.get(key)
        return (time.time() - t) if t is not None else float("inf")

    def _set(self, key: str, val: Any) -> None:
        with self._lock:
            self._store[key] = val
            self._ts[key] = time.time()

    @property
    def collision_seq(self) -> int:
        with self._lock:
            return self._collision_seq

    # ---- RX loop -----------------------------------------------------------
    def _rx_loop(self) -> None:
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except (ConnectionResetError, OSError):
                time.sleep(0.01)
                continue
            if msg is None:
                time.sleep(0.001)
                continue
            t = msg.get_type()
            if t == "BAD_DATA":
                continue
            handler = self._handlers.get(t)
            if handler:
                handler(self, msg)

    def _on_attitude(self, msg) -> None:
        self._set("attitude", {
            "roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
            "rollspeed": msg.rollspeed, "pitchspeed": msg.pitchspeed,
            "yawspeed": msg.yawspeed, "ts": time.time_ns(),
        })

    def _on_imu(self, msg) -> None:
        self._set("highres_imu", {
            "xacc": msg.xacc, "yacc": msg.yacc, "zacc": msg.zacc,
            "xgyro": msg.xgyro, "ygyro": msg.ygyro, "zgyro": msg.zgyro,
            "pressure_alt": msg.pressure_alt, "temperature": msg.temperature,
            "ts_us": msg.time_usec,
        })

    def _on_local_position(self, msg) -> None:  # ABSENT in VQ2; PRIV if present
        self._set("local_position_ned", {
            "x": msg.x, "y": msg.y, "z": msg.z,
            "vx": msg.vx, "vy": msg.vy, "vz": msg.vz, "ts": time.time_ns(),
        })

    def _on_odometry(self, msg) -> None:  # ABSENT in VQ2; PRIV if present
        self._set("odometry", {
            "x": msg.x, "y": msg.y, "z": msg.z,
            "vx": msg.vx, "vy": msg.vy, "vz": msg.vz, "ts": time.time_ns(),
        })

    def _on_collision(self, msg) -> None:
        with self._lock:
            self._collision_seq += 1
            seq = self._collision_seq
        label = {1001: "Gate", 1002: "Environment"}.get(msg.id, str(msg.id))
        self._set("collision", {
            "id": msg.id, "label": label, "threat": msg.threat_level,
            "impulse": msg.horizontal_minimum_delta, "seq": seq, "ts": time.time_ns(),
        })

    def _on_actuator(self, msg) -> None:
        self._set("actuator", {"out": list(msg.actuator[:4])})

    def _on_encapsulated(self, msg) -> None:
        raw = bytes(msg.data)
        if not raw:
            return
        if raw[0] == ENCAP_RACE_STATUS:
            self._on_race_status(raw)
        elif raw[0] == ENCAP_TRACK_INFO:
            self._on_track_packet(msg, raw)

    def _on_race_status(self, raw: bytes) -> None:
        (_type, sim_boot_ms, race_start_ms, race_finish_ns,
         active_gate_idx, last_gate_time) = struct.unpack_from("<BQqqIq", raw)
        self._set("race_status", {
            "sim_boot_ms": sim_boot_ms, "race_start_ms": race_start_ms,
            "race_finish_ns": race_finish_ns, "active_gate": active_gate_idx,
            "last_gate_time": last_gate_time, "ts": time.time_ns(),
        })

    def _on_handshake(self, msg) -> None:
        transfer_id = msg.width
        self._track_chunks[transfer_id] = {}
        self._expected_chunks[transfer_id] = msg.packets

    def _on_track_packet(self, msg, raw: bytes) -> None:
        _type, transfer_id = struct.unpack_from("<BH", raw)
        if transfer_id not in self._expected_chunks:
            return
        self._track_chunks[transfer_id][msg.seqnr] = raw[3:]
        if len(self._track_chunks[transfer_id]) == self._expected_chunks[transfer_id]:
            full = b"".join(self._track_chunks[transfer_id][i]
                            for i in range(len(self._track_chunks[transfer_id])))
            del self._track_chunks[transfer_id]
            del self._expected_chunks[transfer_id]
            self._parse_track(full)

    def _parse_track(self, payload: bytes) -> None:
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            (gid, x, y, z, qw, qx, qy, qz, w, h) = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates.append({"id": gid, "pos": (x, y, z), "quat": (qw, qx, qy, qz),
                          "size": (w, h)})
        self._set("track_gates", gates)

    _handlers = {
        "ATTITUDE": _on_attitude,
        "HIGHRES_IMU": _on_imu,
        "LOCAL_POSITION_NED": _on_local_position,
        "ODOMETRY": _on_odometry,
        "COLLISION": _on_collision,
        "ACTUATOR_OUTPUT_STATUS": _on_actuator,
        "ENCAPSULATED_DATA": _on_encapsulated,
        "DATA_TRANSMISSION_HANDSHAKE": _on_handshake,
    }

    # ---- TX ----------------------------------------------------------------
    def send_attitude_target(self, thrust: float, roll_rate: float,
                             pitch_rate: float, yaw_rate: float) -> None:
        now_ms = int(time.time() * 1000)
        self.conn.mav.set_attitude_target_send(
            now_ms - self.boot_ms,
            self.conn.target_system, self.conn.target_component,
            _ATTITUDE_IGNORE, [1, 0, 0, 0],
            roll_rate, pitch_rate, yaw_rate, thrust,
        )

    def arm(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def send_reset(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            self.reset_cmd_id, 0, 0, 0, 0, 0, 0, 0, 0,
        )

    def send_timesync(self) -> None:
        self.conn.mav.timesync_send(int(time.time_ns()), 0)
