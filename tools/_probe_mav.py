"""Quick MAVLink probe — is the race publishing?"""
from pymavlink import mavutil
import time

c = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
t0 = time.time()
types = {}
while time.time() - t0 < 2.5:
    m = c.recv_match(blocking=True, timeout=0.5)
    if m:
        types[m.get_type()] = types.get(m.get_type(), 0) + 1
print(types if types else 'NO PACKETS — start race in 3391 sim first')
