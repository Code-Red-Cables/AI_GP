import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False
        self.track_chunks = {}
        self.expected_num_track_chunks = {}
        self._last_gate_idx = None

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(target=rx.mavlink_receive_loop, daemon=False)
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _log(self, event, detail=''):
        fn = self.data.get('log_event')
        if fn:
            fn(event, detail)

    # ------------------------------------------------------------------
    def mavlink_receive_loop(self):
        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError — no longer listening.', flush=True)
                return

            if msg is None:
                time.sleep(0.001)
                continue

            t = msg.get_type()
            if t == 'BAD_DATA':
                continue

            if   t == 'HEARTBEAT':              self.on_heartbeat(msg)
            elif t == 'TIMESYNC':               self.on_timesync(msg)
            elif t == 'ATTITUDE':               self.on_attitude(msg)
            elif t == 'LOCAL_POSITION_NED':     self.on_local_position_ned(msg)
            elif t == 'ODOMETRY':               self.on_odometry(msg)
            elif t == 'HIGHRES_IMU':            self.on_highres_imu(msg)
            elif t == 'ENCAPSULATED_DATA':      self.on_encapsulated_data(msg)
            elif t == 'ACTUATOR_OUTPUT_STATUS': self.on_actuator_output_status(msg)
            elif t == 'COLLISION':              self.on_collision(msg)
            elif t == 'DATA_TRANSMISSION_HANDSHAKE':
                transfer_id = msg.width
                self.track_chunks[transfer_id] = {}
                self.expected_num_track_chunks[transfer_id] = msg.packets

    # ------------------------------------------------------------------
    def on_heartbeat(self, msg):
        pass

    def on_timesync(self, msg):
        pass

    def on_attitude(self, msg):
        # NOTE: ATTITUDE is marked disabled in VQ2 spec but still arrives.
        # This is our only yaw source (AHRS has no magnetometer in sim).
        self.data['attitude'] = {
            'roll':       msg.roll,
            'pitch':      msg.pitch,
            'yaw':        msg.yaw,
            'rollspeed':  msg.rollspeed,
            'pitchspeed': msg.pitchspeed,
            'yawspeed':   msg.yawspeed,
            'ts':         time.time_ns(),
        }

    def on_local_position_ned(self, msg):
        # Disabled in VQ2 — stored here for VQ1 compatibility.
        self.data['local_position_ned'] = {
            'x': msg.x, 'y': msg.y, 'z': msg.z,
            'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
            'ts': time.time_ns(),
        }

    def on_odometry(self, msg):
        # Disabled in VQ2 — stored here for VQ1 compatibility.
        self.data['odometry'] = {
            'x': msg.x, 'y': msg.y, 'z': msg.z,
            'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
            'ts': time.time_ns(),
        }

    def on_highres_imu(self, msg):
        self.data['highres_imu'] = {
            'xacc': msg.xacc, 'yacc': msg.yacc, 'zacc': msg.zacc,
            'xgyro': msg.xgyro, 'ygyro': msg.ygyro, 'zgyro': msg.zgyro,
            'xmag': msg.xmag, 'ymag': msg.ymag, 'zmag': msg.zmag,
            'abs_pressure': msg.abs_pressure,
            'diff_pressure': msg.diff_pressure,
            'pressure_alt': msg.pressure_alt,
            'temperature': msg.temperature,
            'ts_us': msg.time_usec,
            'ts': time.time_ns(),
        }

    def on_encapsulated_data(self, msg):
        if not msg:
            return
        raw = bytes(msg.data)
        data_type = raw[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self.on_race_status(msg)
        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw = bytes(msg.data)
        (data_type, sim_boot_ms, race_start_ms, race_finish_ns,
         active_gate_idx, last_gate_time) = struct.unpack_from('<BQqqIq', raw)

        self.data['race_status'] = {
            'sim_boot_ms':    sim_boot_ms,
            'race_start_ms':  race_start_ms,
            'race_finish_ns': race_finish_ns,
            'active_gate':    active_gate_idx,
            'last_gate_time': last_gate_time,
        }

        if active_gate_idx != self._last_gate_idx:
            if self._last_gate_idx is not None:
                self._log('GATE_PASSED',
                          f'gate={active_gate_idx}  last_gate_time={last_gate_time/1e9:.3f}s')
            self._last_gate_idx = active_gate_idx

        if race_finish_ns > 0:
            self._log('RACE_FINISH',
                      f'finish_time={race_finish_ns/1e9:.3f}s')

    def on_track_data_packet(self, msg):
        raw = bytes(msg.data)
        data_type, transfer_id = struct.unpack_from('<BH', raw)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw = raw[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full += self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full)

    def on_track_data(self, payload):
        # Gate position/orientation nulled in VQ2 per spec.
        num_gates, = struct.unpack_from('<H', payload)
        payload = payload[2:]
        gates = []
        for i in range(num_gates):
            (gate_id, x, y, z, qw, qx, qy, qz, w, h) = struct.unpack_from(
                '<Hfffffffff', payload)
            payload = payload[38:]
            gates.append({'id': gate_id, 'pos': (x, y, z), 'size': (w, h)})
        self.data['track_gates'] = gates
        self._log('TRACK_DATA', f'{num_gates} gates received')

    def on_actuator_output_status(self, msg):
        self.data['actuator'] = {
            'fl': msg.actuator[0],
            'fr': msg.actuator[1],
            'bl': msg.actuator[2],
            'br': msg.actuator[3],
        }

    def on_collision(self, msg):
        # collision_id: 1001=Gate, 1002=Environment
        label = {1001: 'Gate', 1002: 'Environment'}.get(msg.id, str(msg.id))
        self.data['collision'] = {
            'id': msg.id, 'label': label,
            'threat': msg.threat_level,
            'impulse': msg.horizontal_minimum_delta,
            'ts': time.time_ns(),
        }
        self._log('COLLISION',
                  f'{label}  threat={msg.threat_level}  impulse={msg.horizontal_minimum_delta:.2f}kg·m/s')
