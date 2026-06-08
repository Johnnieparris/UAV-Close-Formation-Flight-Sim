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

        # --- TURN & THROTTLE COMPENSATION (FEED-FORWARD) ---
        self.K_FEEDFORWARD = 0.8 
        self.K_PITCH_FF = 0.7
        self.K_THRUST_FF = 0.4
        
        
        self.MAX_ROLL = math.radians(40)   
        self.MAX_PITCH = math.radians(15)  
        
        # Command Smoothing (Low-Pass Filter)
        self.last_roll_cmd = 0.0
        self.last_pitch_cmd = 0.0
        self.SMOOTHING_ALPHA = 1

        # State tracking
        self.leader_ned = self._empty_ned()
        self.follower_ned = self._empty_ned()
        self.leader_roll = 0.0
        self.leader_pitch = 0.0
        self.leader_throttle = 0.0  # Normalized (0.0 to 1.0)
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
        
        # Request Position, Attitude, and HUD data streams
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000) 
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 50_000)  # For true throttle
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
        msg_type = msg.get_type()
        if msg_type == 'LOCAL_POSITION_NED':
            self.leader_ned['n'], self.leader_ned['e'], self.leader_ned['d'] = msg.x, msg.y, msg.z
            self.leader_ned['valid'] = True
            speed = math.hypot(msg.vx, msg.vy)
            if speed > 0.5:
                self.chi_L = math.atan2(msg.vy, msg.vx)
        elif msg_type == 'ATTITUDE':
            self.leader_roll = msg.roll
            self.leader_pitch = msg.pitch
            self.chi_L = msg.yaw
        elif msg_type == 'VFR_HUD':
            # msg.throttle is 0 to 100 integer percent; scale to 0.0 - 1.0 float
            self.leader_throttle = msg.throttle / 100.0

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
        print("Running predictive loop with Feed-Forward turn & leader throttle compensation.\n")

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
                    heading_to_slot = math.atan2(slot_e - f_e, slot_n - f_n)
                    heading_to_slot = (heading_to_slot + math.pi) % (2 * math.pi) - math.pi

                    if abs(self.leader_roll) < math.radians(5.0):
                        MAX_INTERCEPT_ANGLE = math.radians(25) 
                        heading_correction = cross_err * self.K_CROSS_HEADING 
                        heading_correction = max(-MAX_INTERCEPT_ANGLE, min(MAX_INTERCEPT_ANGLE, heading_correction))
                        target_yaw = self.chi_L + heading_correction
                    else:
                        blend_factor = min(1.0, dist_to_slot / 30.0)
                        diff = heading_to_slot - self.chi_L
                        diff = (diff + math.pi) % (2 * math.pi) - math.pi
                        target_yaw = self.chi_L + (diff * blend_factor)

                    target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi

                    heading_error = target_yaw - self.chi_F
                    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

                    feedback_roll = heading_error * self.K_ROLL
                    feedforward_roll = self.K_FEEDFORWARD * self.leader_roll

                    raw_roll = feedback_roll + feedforward_roll
                    raw_roll = max(-self.MAX_ROLL, min(self.MAX_ROLL, raw_roll))
                    # ---------------------------------------------------------

                    # 3. Pitch Calculations
                    alt_error = f['d'] - slot_d 
                    feedback_pitch = alt_error * self.K_PITCH    

                    feedforward_pitch = self.K_PITCH_FF * self.leader_pitch
                    raw_pitch = feedback_pitch + feedforward_pitch
                    raw_pitch = max(-self.MAX_PITCH, min(self.MAX_PITCH, raw_pitch))               

                    # 4. Command Low-Pass Filtering & Dynamic Thrust Calculations
                    smoothed_roll = (self.SMOOTHING_ALPHA * raw_roll) + ((1.0 - self.SMOOTHING_ALPHA) * self.last_roll_cmd)
                    smoothed_pitch = (self.SMOOTHING_ALPHA * raw_pitch) + ((1.0 - self.SMOOTHING_ALPHA) * self.last_pitch_cmd)
                    
                    # Core distance feedback
                    feedback_thrust = along_err * self.K_THRUST      
                    
                    # Turn lift-loss compensation
                    feedforward_trust = (1.0 / math.cos(smoothed_roll) - 1.0) * self.K_THRUST_FF

                    # True Throttle Feed-Forward: Replaces static self.BASE_THRUST

                    # Combine everything
                    target_thrust = self.BASE_THRUST + feedback_thrust + feedforward_trust
                    target_thrust = max(0.0, min(1.0, target_thrust)) 

                    self.last_roll_cmd, self.last_pitch_cmd = smoothed_roll, smoothed_pitch

                    # Send payload
                    self.send_attitude_target(smoothed_roll, smoothed_pitch, target_yaw, target_thrust)

                    # 5. Updated Telemetry Diagnostics Log
                    now = time.monotonic()
                    if now - self._last_log_time >= self.log_interval_s:
                        self._last_log_time = now
                        print(f"Along:{along_err:+5.1f}m Cross:{cross_err:+5.1f}m | "
                              f"L_Thrust:{self.leader_throttle*100:3.0f}% F_Thrust:{target_thrust*100:3.0f}% | "
                              f"L_P:{math.degrees(self.leader_pitch):+05.1f}° F_P:{math.degrees(smoothed_pitch):+05.1f}° | "
                              f"L_A:{-l_d:+06.1f} F_A:{-f['d']:+06.1f}")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nShutting down controller.")

if __name__ == "__main__":
    clearances = {'f_c': 10.0, 'l_c': 0.0, 'v_c': 0.0}
    controller = ArduPlaneFeedForwardController("udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances)
    controller.run()