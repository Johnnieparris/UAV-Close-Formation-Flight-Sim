import time
import math
from pymavlink import mavutil
import numpy as np

# Ignore roll/pitch/yaw body rates; target attitude quaternion & thrust only.
_ATTITUDE_ONLY_TYPE_MASK = 0b00000111

class CascadedFormationController:
    def __init__(self, leader_port, follower_port, offsets):
        print(f"Connecting to Leader on {leader_port}...")
        self.leader = mavutil.mavlink_connection(leader_port)

        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)

        # Formation geometry offsets (Meters)
        self.f_c = offsets.get('f_c', 0.0) 
        self.l_c = offsets.get('l_c', 0.0) 
        self.v_c = offsets.get('v_c', 0.0) 

        # --- PHYSICAL AIRCRAFT PARAMETERS 
        
        self.AIRCRAFT_MASS = 5.9       
        self.WING_AREA = 0.982         
        self.AIR_DENSITY = 1.225       # kg/m^3 (Sea level standard)
        self.CD_0 = 0.028               # Parasitic drag coefficient
        self.CD_ALPHA = 0.04           # Induced drag coefficient multiplier
        self.CL_0 = 0.25      
        self.CL_ALPHA = 5.0    
        self.MAX_THRUST_NEWTONS = 38.9 # Max engine thrust in Newtons
        self.GRAVITY = 9.81            # m/s^2
                
        # State tracking
        self.leader_ned = self._empty_ned()
        self.follower_ned = self._empty_ned()
        self.leader_roll = 0.0  
        self.has_offset = False
        self.offset = {'n': 0.0, 'e': 0.0, 'd': 0.0}
        self.chi_L = 0.0  
        self.chi_F = 0.0  
        self.dt = 0.05
        
        self._last_log_time = 0.0
        self.log_interval_s = 0.5 

        # Wait for connections
        self.leader.wait_heartbeat()
        print("Leader heartbeat verified.")
        self.follower.wait_heartbeat()
        print("Follower heartbeat verified.")
        
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000) 
        self._request_message_stream(self.follower, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)

    @staticmethod
    def _empty_ned():
        return {'n': 0.0, 'e': 0.0, 'd': 0.0, 'vn': 0.0, 've': 0.0, 'vd': 0.0, 'valid': False}

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
        mode_id = self.follower.mode_mapping().get(mode_name)
        if mode_id is not None:
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
            self.leader_ned['vn'], self.leader_ned['ve'], self.leader_ned['vd'] = msg.vx, msg.vy, msg.vz
            self.leader_ned['valid'] = True
            

        elif msg.get_type() == 'ATTITUDE':
            self.leader_roll = msg.roll
            self.chi_L = msg.yaw  

    def _handle_follower_msg(self, msg):
        if msg.get_type() == 'LOCAL_POSITION_NED':
            self.follower_ned['n'], self.follower_ned['e'], self.follower_ned['d'] = msg.x, msg.y, msg.z
            self.follower_ned['vn'], self.follower_ned['ve'], self.follower_ned['vd'] = msg.vx, msg.vy, msg.vz
            self.follower_ned['valid'] = True
        
        elif msg.get_type() == 'ATTITUDE':
            self.follower_roll = msg.roll
            self.chi_F = msg.yaw  

    def _sync_offset(self):
        self.offset = {'n': 0.0, 'e': 0.0, 'd': 0.0} 
        self.has_offset = True

    def send_attitude_target(self, roll, pitch, yaw, thrust):
        q = self.euler_to_quaternion(roll, pitch, yaw)
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
            print("Running Cascaded Controller: Kinematic Outer Loop + NLDI Inner Loop\n")

            last_time = time.monotonic()

            try:
                while True:
                    now = time.monotonic()
                    self.dt = max(0.001, now - last_time)
                    last_time = now

                    self._drain_messages(self.leader, self._handle_leader_msg)
                    self._drain_messages(self.follower, self._handle_follower_msg)

                    if self.has_offset:
                        l_n, l_e, l_d = self.leader_ned['n'], self.leader_ned['e'], self.leader_ned['d']
                        
                        # Follower positions
                        f_n = self.follower_ned['n'] + self.offset['n']
                        f_e = self.follower_ned['e'] + self.offset['e']
                        f_d = self.follower_ned['d'] + self.offset['d']

                        # ==========================================
                        # 1. OUTER LOOP NLDI
                        # ==========================================
                        
                        cos_chi_L = math.cos(self.chi_L)
                        sin_chi_L = math.sin(self.chi_L)
                        
                        d_n = self.f_c * cos_chi_L + self.l_c * sin_chi_L
                        d_e = self.f_c * sin_chi_L - self.l_c * cos_chi_L
                        d_d = self.v_c
                        
                        # Target 3D position
                        pd_n = l_n - d_n
                        pd_e = l_e - d_e
                        pd_d = l_d + d_d
                        
                        # --- 3D Error Calculation ---
                        e_n = f_n - pd_n
                        e_e = f_e - pd_e
                        e_d = f_d - pd_d
                        
                        # Target 3D velocity (Approximated to leader velocity)
                        vd_n, vd_e, vd_d = self.leader_ned['vn'], self.leader_ned['ve'], self.leader_ned['vd']
                        
                        # Velocity Error calculation
                        v_n, v_e, v_d = self.follower_ned['vn'], self.follower_ned['ve'], self.follower_ned['vd']
                        edot_n = v_n - vd_n
                        edot_e = v_e - vd_e
                        edot_d = v_d - vd_d

                        # --- Outer Loop NLDI: Commanded Accelerations ---
                        # Controller gains (Tune these for responsiveness)
                        K_P = np.array([
                            [0.00001, 0.0, 0.0],  # X-axis (Forward): Highly responsive
                            [0.0, 0.8, 0.0],  # Y-axis (Lateral): Highly responsive
                            [0.0, 0.0, 10]   # Z-axis (Vertical): Lower gain to prevent aggressive pitch/stall
                        ])
                        
                        K_D = np.array([
                            [0, 0.0, 0.0],  # X-axis damping
                            [0.0, 0, 0.0],  # Y-axis damping
                            [0.0, 0.0, 10]   # Z-axis damping (tuned to match lower proportional gain)
                        ])
                        
                        # --- Vectorize Errors ---
                        # Pack the individual errors into 3D numpy arrays
                        e_vec = np.array([e_n, e_e, e_d])
                        edot_vec = np.array([edot_n, edot_e, edot_d])

                        # --- Matrix Multiplication ---
                        # Calculate the 3D acceleration command vector using the '@' matrix multiplication operator
                        a_cmd_vec = -K_P @ e_vec - K_D @ edot_vec
                        
                        # --- Unpack Commands ---
                        # Extract the individual commanded accelerations for the inner loop
                        a_n_cmd = a_cmd_vec[0]
                        a_e_cmd = a_cmd_vec[1]
                        a_d_cmd = a_cmd_vec[2]

                        # --- Inner Loop Mapping (Acceleration -> Attitude/Thrust) ---
                        # Calculate follower's current airspeed and path angles

                        V = math.sqrt(v_n**2 + v_e**2 + v_d**2)
                        if V < 3.0: V = 3.0 # Prevent division by zero / singularity on the ground

                        gamma = math.asin(-v_d / V) # Flight path angle
                        chi_F = math.atan2(v_e, v_n) # Follower Azimuth

                        # 1. Roll Command (from Lateral Acceleration)
                        a_lat_cmd = math.cos(chi_F) * a_e_cmd - math.sin(chi_F) * a_n_cmd
                        # Simplification of phi = arctan(m*a_lat*v / L) assuming L ~= mg
                        roll_cmd = math.atan2(a_lat_cmd, self.GRAVITY)
                        roll_cmd = max(-math.radians(45), min(math.radians(45), roll_cmd)) 


                        # 2. Pitch Command (from Vertical Acceleration)
                        # Lift required to achieve commanded Z acceleration (Positive Z is down)
                        lift_cmd = self.AIRCRAFT_MASS * (self.GRAVITY - a_d_cmd) / math.cos(roll_cmd)
                        
                        # Aerodynamic lift conversion
                        q_bar = 0.5 * self.AIR_DENSITY * (V**2)
                        CL_req = lift_cmd / (q_bar * self.WING_AREA)
                        alpha_cmd = (CL_req - self.CL_0) / self.CL_ALPHA
                        
                        pitch_cmd = gamma + alpha_cmd
                        pitch_cmd = max(-math.radians(20), min(math.radians(25), pitch_cmd)) # Limit pitch

                        # 3. Throttle Command (from Forward Acceleration)
                        a_forward_cmd = (v_n * a_n_cmd + v_e * a_e_cmd + v_d * a_d_cmd) / V
                        
                        # Estimate total drag
                        drag = q_bar * self.WING_AREA * (self.CD_0 + self.CD_ALPHA * (alpha_cmd**2))
                        
                        # Thrust required
                        thrust_cmd = self.AIRCRAFT_MASS * (a_forward_cmd + self.GRAVITY * math.sin(gamma)) + drag
                        throttle_cmd = thrust_cmd / self.MAX_THRUST_NEWTONS
                        throttle_cmd = max(0.0, min(1.0, throttle_cmd)) # Bound [0, 1]

                        # Project errors into leader body frame (along / cross / down)
                        along_err = e_n * cos_chi_L + e_e * sin_chi_L
                        cross_err = -e_n * sin_chi_L + e_e * cos_chi_L

                        # Saturation flags (True = hit limit — NLDI output clipped)
                        roll_sat = abs(roll_cmd) >= math.radians(45) - 1e-6
                        pitch_sat = pitch_cmd <= math.radians(-20) + 1e-6 or pitch_cmd >= math.radians(25) - 1e-6
                        thr_sat = throttle_cmd <= 1e-6 or throttle_cmd >= 1.0 - 1e-6
                        v_floor = math.sqrt(v_n**2 + v_e**2 + v_d**2) < 3.0

                        self.send_attitude_target(roll_cmd, pitch_cmd, 0, throttle_cmd)

                        # --- NLDI diagnostic dump (each stage of the cascade) ---
                        if now - self._last_log_time >= self.log_interval_s:
                            self._last_log_time = now
                            V_raw = math.sqrt(v_n**2 + v_e**2 + v_d**2)
                            print(
                                f"── NLDI dt={self.dt*1000:.0f}ms ──\n"
                                f"  POS err  along={along_err:+7.1f}m  cross={cross_err:+7.1f}m  down={e_d:+7.1f}m\n"
                                f"  VEL err  N={edot_n:+6.2f}  E={edot_e:+6.2f}  D={edot_d:+6.2f}  m/s\n"
                                f"  OUTER a* N={a_n_cmd:+6.2f}  E={a_e_cmd:+6.2f}  D={a_d_cmd:+6.2f}  m/s²"
                                f"  (Kp={K_P} Kd={K_D})\n"
                                f"  KIN     V={V_raw:5.1f}→{V:5.1f}m/s  γ={math.degrees(gamma):+5.1f}°"
                                f"  χF={math.degrees(chi_F):+6.1f}°  χL={math.degrees(self.chi_L):+6.1f}°"
                                f"{'  [V FLOOR]' if v_floor else ''}\n"
                                f"  ROLL    a_lat*={a_lat_cmd:+6.2f}m/s²  → φ*={math.degrees(roll_cmd):+5.1f}°"
                                f"{'  [SAT]' if roll_sat else ''}\n"
                                f"  PITCH   L*={lift_cmd:6.1f}N  CL*={CL_req:+5.2f}  α*={math.degrees(alpha_cmd):+5.1f}°"
                                f"  → θ*={math.degrees(pitch_cmd):+5.1f}°"
                                f"{'  [SAT]' if pitch_sat else ''}\n"
                                f"  THRUST  a_fwd*={a_forward_cmd:+6.2f}m/s²  D={drag:5.1f}N"
                                f"  T*={thrust_cmd:6.1f}N  → thr={throttle_cmd:4.2f}"
                                f"{'  [SAT]' if thr_sat else ''}\n"
                                f"  SEND    φ={math.degrees(roll_cmd):+5.1f}°"
                                f"  θ={math.degrees(pitch_cmd):+5.1f}°"
                                f"  ψ=0.0° (hardcoded)  thr={throttle_cmd:4.2f}\n"
                            )

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("\nShutting down controller.")

if __name__ == "__main__":
    clearances = {'f_c': 10.0, 'l_c': 5.0, 'v_c': 0}

    controller = CascadedFormationController("udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances)
    controller.run()
