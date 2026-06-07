import time
import math
from pymavlink import mavutil

# Ignore roll/pitch/yaw body rates; target attitude quaternion & thrust only.
_ATTITUDE_ONLY_TYPE_MASK = 0b00000111

class ArduPlaneFeedForwardController:
    def __init__(self, leader_port, follower_port, offsets):
        print(f"Connecting to Leader on {leader_port}...")
        self.leader = mavutil.mavlink_connection(leader_port)

        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)

        # Formation geometry offsets
        self.f_c = offsets.get('f_c', 0.0) 
        self.l_c = offsets.get('l_c', 0.0) 
        self.v_c = offsets.get('v_c', 0.0) 

        # --- TUNED CONTROLLER GAINS ---
        self.K_ROLL = 0.4
        self.K_PITCH = 0.02
        self.K_THRUST = 0.006
        self.BASE_THRUST = 0.0834
        self.K_CROSS_HEADING = 0.02

        # --- TURN COMPENSATION (FEED-FORWARD) ---
        self.K_FEEDFORWARD = 0.8 
        
        self.MAX_ROLL = math.radians(40)   
        self.MAX_PITCH = math.radians(10)  
        
        # Command Smoothing (Low-Pass Filter)
        self.last_roll_cmd = 0.0
        self.last_pitch_cmd = 0.0
        self.SMOOTHING_ALPHA = 1

        # State tracking
        self.leader_ned = self._empty_ned()
        self.follower_ned = self._empty_ned()
        self.leader_roll = 0.0  
        self.has_offset = False
        self.offset = {'n': 0.0, 'e': 0.0, 'd': 0.0}
        
        self.chi_L = 0.0  
        self.chi_F = 0.0  

        self._last_log_time = 0.0
        self.log_interval_s = 0.5 

        self.leader.wait_heartbeat()
        print("Leader heartbeat verified.")
        self.follower.wait_heartbeat()
        print("Follower heartbeat verified.")
        
        # Request both Position and Attitude streams
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000) 
        self._request_message_stream(self.follower, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)

    @staticmethod
    def _empty_ned():
        return {'n': 0.0, 'e': 0.0, 'd': 0.0, 'vn': 0.0, 've': 0.0, 'valid': False}

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ]

    def _request_message_stream(self, link, msg_id, interval_us):
        link.mav.command_long_send(
            link.target_system, link.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, interval_us, 0, 0, 0, 0, 0,
        )

    def set_follower_mode(self, mode_name):
        if mode_name not in self.follower.mode_mapping():
            print(f"Unknown mode: {mode_name}")
            return
        mode_id = self.follower.mode_mapping()[mode_name]
        self.follower.mav.set_mode_send(
            self.follower.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

    def _drain_messages(self, link, handler):
        while True:
            msg = link.recv_msg()
            if msg is None:
                break
            handler(msg)

    def _handle_leader_msg(self, msg):
        if msg.get_type() == 'LOCAL_POSITION_NED':
            self.leader_ned['n'], self.leader_ned['e'], self.leader_ned['d'] = msg.x, msg.y, msg.z
            self.leader_ned['valid'] = True
            speed = math.hypot(msg.vx, msg.vy)
            if speed > 0.5:
                self.chi_L = math.atan2(msg.vy, msg.vx)
        elif msg.get_type() == 'ATTITUDE':
            self.leader_roll = msg.roll
            self.chi_L = msg.yaw  

    def _handle_follower_msg(self, msg):
        if msg.get_type() == 'LOCAL_POSITION_NED':
            self.follower_ned['n'], self.follower_ned['e'], self.follower_ned['d'] = msg.x, msg.y, msg.z
            self.follower_ned['valid'] = True
            speed = math.hypot(msg.vx, msg.vy)
            if speed > 0.5:
                self.chi_F = math.atan2(msg.vy, msg.vx)

    def _sync_offset(self):
        self.offset = {'n': 0.0, 'e': 0.0, 'd': 0.0} 
        self.has_offset = True

    def _follower_in_leader_ned(self):
        return {
            'n': self.follower_ned['n'] + self.offset['n'],
            'e': self.follower_ned['e'] + self.offset['e'],
            'd': self.follower_ned['d'] + self.offset['d'],
        }

    def formation_separation(self):
        l_n, l_e = self.leader_ned['n'], self.leader_ned['e']
        f = self._follower_in_leader_ned()
        f_n, f_e = f['n'], f['e']
        cos_c, sin_c = math.cos(self.chi_L), math.sin(self.chi_L)
        dn, de = l_n - f_n, l_e - f_e
        return (dn * cos_c + de * sin_c), (-dn * sin_c + de * cos_c)

    def send_attitude_target(self, roll, pitch, yaw, thrust):
        q = self.euler_to_quaternion(roll, pitch, 0)
        self.follower.mav.set_attitude_target_send(
            0, self.follower.target_system, self.follower.target_component,
            _ATTITUDE_ONLY_TYPE_MASK, q, 0, 0, 0, thrust
        )

    def run(self):
        print("Syncing frames...")
        while not (self.has_offset and self.leader_ned['valid'] and self.follower_ned['valid']):
            self._drain_messages(self.leader, self._handle_leader_msg)
            self._drain_messages(self.follower, self._handle_follower_msg)
            if not self.has_offset and self.leader_ned['valid'] and self.follower_ned['valid']:
                self._sync_offset()
            time.sleep(0.05)
            
        self.set_follower_mode('GUIDED')
        print("Running predictive loop with Feed-Forward turn compensation.\n")

        try:
            while True:
                self._drain_messages(self.leader, self._handle_leader_msg)
                self._drain_messages(self.follower, self._handle_follower_msg)

                if self.has_offset:
                    l_n, l_e, l_d = self.leader_ned['n'], self.leader_ned['e'], self.leader_ned['d']
                    f = self._follower_in_leader_ned()
                    f_n, f_e = f['n'], f['e']

                    # 1. Target Tracking Slot Calculations
                    cos_c, sin_c = math.cos(self.chi_L), math.sin(self.chi_L)
                    slot_n = l_n - (self.f_c * cos_c + self.l_c * sin_c)
                    slot_e = l_e - (self.f_c * sin_c - self.l_c * cos_c)
                    slot_d = l_d + self.v_c

                    dist_to_slot = math.hypot(slot_n - f_n, slot_e - f_e)

                    along_m, cross_m = self.formation_separation()
                    cross_err = cross_m - self.l_c
                    along_err = along_m - self.f_c

                    # ---------------------------------------------------------
                    # 2. Roll Calculation (Vector Pursuit Geometry)
                    # ---------------------------------------------------------
                    # Calculate the baseline heading directly toward the slot
                    heading_to_slot = math.atan2(slot_e - f_e, slot_n - f_n)
                    heading_to_slot = (heading_to_slot + math.pi) % (2 * math.pi) - math.pi

                    # If the leader is flying straight, we want to match their course 
                    # and use cross-track error to slide left/right.
                    if abs(self.leader_roll) < math.radians(5.0):
                        MAX_INTERCEPT_ANGLE = math.radians(25) 
                        heading_correction = cross_err * self.K_CROSS_HEADING 
                        heading_correction = max(-MAX_INTERCEPT_ANGLE, min(MAX_INTERCEPT_ANGLE, heading_correction))
                        target_yaw = self.chi_L + heading_correction
                    
                    # If the leader is actively turning, matching their heading causes corner cutting.
                    # Instead, force the follower to steer directly toward the spatial slot coordinates.
                    else:
                        # Blend the leader's heading with the direct line to the slot based on distance.
                        # This creates an "elastic tether" to the slot position.
                        blend_factor = min(1.0, dist_to_slot / 30.0) # 0.0 at slot, 1.0 when 30m away
                        
                        # Unroll/unwrap angles safely for linear interpolation
                        diff = heading_to_slot - self.chi_L
                        diff = (diff + math.pi) % (2 * math.pi) - math.pi
                        
                        target_yaw = self.chi_L + (diff * blend_factor)

                    # Normalize target_yaw
                    target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi

                    # Core Heading Tracking Loop
                    heading_error = target_yaw - self.chi_F
                    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

                    feedback_roll = heading_error * self.K_ROLL
                    
                    feedforward_roll = self.K_FEEDFORWARD * self.leader_roll

                    raw_roll = feedback_roll + feedforward_roll
                    raw_roll = max(-self.MAX_ROLL, min(self.MAX_ROLL, raw_roll))
                    # ---------------------------------------------------------

                    # 3. Pitch and Thrust Calculations
                    alt_error = f['d'] - slot_d 
                    raw_pitch = max(-self.MAX_PITCH, min(self.MAX_PITCH, alt_error * self.K_PITCH))


                    target_thrust = self.BASE_THRUST + (along_err * self.K_THRUST)
                    

                    # 4. Command Low-Pass Filtering
                    smoothed_roll = (self.SMOOTHING_ALPHA * raw_roll) + ((1.0 - self.SMOOTHING_ALPHA) * self.last_roll_cmd)
                    smoothed_pitch = (self.SMOOTHING_ALPHA * raw_pitch) + ((1.0 - self.SMOOTHING_ALPHA) * self.last_pitch_cmd)
                    
                    self.last_roll_cmd, self.last_pitch_cmd = smoothed_roll, smoothed_pitch

                    # Send payload
                    self.send_attitude_target(smoothed_roll, smoothed_pitch, target_yaw, target_thrust)

                    # 5. Updated Telemetry Diagnostics Log
                    now = time.monotonic()
                    if now - self._last_log_time >= self.log_interval_s:
                        self._last_log_time = now
                        print(f"Follow Err: {along_err:+5.1f}m | "
                              f"Cross-Track Err: {cross_err:+5.1f}m | "
                              f"L_Roll: {math.degrees(self.leader_roll):+05.1f}° | "
                              f"F_Roll: {math.degrees(smoothed_roll):+05.1f}°")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nShutting down controller.")

if __name__ == "__main__":
    clearances = {'f_c': 10.0, 'l_c': 0.0, 'v_c': 0.0}
    controller = ArduPlaneFeedForwardController("udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances)
    controller.run()