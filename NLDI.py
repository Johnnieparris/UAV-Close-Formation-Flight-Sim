import time
import math
from pymavlink import mavutil

import multiprocessing
import matplotlib.pyplot as plt
from collections import deque

def run_visualizer(data_queue):
    """Runs in a separate process to handle real-time plotting."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_along, ax_cross, ax_vert = axes
    fig.suptitle("Outer Loop Velocity Commands vs Follower Response", fontsize=14, color='white')

    # History buffers for trailing traces
    max_history = 100
    time_hist = deque(maxlen=max_history)
    target_along_vel_hist = deque(maxlen=max_history)
    actual_along_vel_hist = deque(maxlen=max_history)
    target_cross_vel_hist = deque(maxlen=max_history)
    actual_cross_vel_hist = deque(maxlen=max_history)
    target_vert_vel_hist = deque(maxlen=max_history)
    actual_vert_vel_hist = deque(maxlen=max_history)

    t_start = time.time()

    while plt.fignum_exists(fig.number):
        # Consume all available data points in the queue to get the latest state
        data = None
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
            except Exception:
                break

        if data is not None:
            # Unpack payload
            target_along_vel = data['target_along_vel']
            actual_along_vel = data['actual_along_vel']
            target_cross_vel = data['target_cross_vel']
            actual_cross_vel = data['actual_cross_vel']
            target_vert_vel = data['target_vert_vel']
            actual_vert_vel = data['actual_vert_vel']
            roll_cmd, throttle_cmd = data['roll_cmd'], data['throttle_cmd']

            # Append command/response histories
            time_hist.append(time.time() - t_start)
            target_along_vel_hist.append(target_along_vel)
            actual_along_vel_hist.append(actual_along_vel)
            target_cross_vel_hist.append(target_cross_vel)
            actual_cross_vel_hist.append(actual_cross_vel)
            target_vert_vel_hist.append(target_vert_vel)
            actual_vert_vel_hist.append(actual_vert_vel)

            time_values = list(time_hist)

            for ax in axes:
                ax.clear()
                ax.grid(True, color='gray', alpha=0.3)

            ax_along.plot(time_values, list(target_along_vel_hist), 'y-', label='Outer Loop Cmd')
            ax_along.plot(time_values, list(actual_along_vel_hist), 'c--', label='Follower Actual')
            ax_along.set_title(f"Along-Track Velocity | Thr Cmd: {throttle_cmd*100:.1f}%")
            ax_along.set_ylabel("Velocity (m/s)")
            ax_along.legend(loc='upper left')

            ax_cross.plot(time_values, list(target_cross_vel_hist), 'y-', label='Outer Loop Cmd')
            ax_cross.plot(time_values, list(actual_cross_vel_hist), 'c--', label='Follower Actual')
            ax_cross.set_title(f"Cross-Track Velocity | Roll Cmd: {math.degrees(roll_cmd):.1f}°")
            ax_cross.set_ylabel("Velocity (m/s)")
            ax_cross.legend(loc='upper left')

            ax_vert.plot(time_values, list(target_vert_vel_hist), 'y-', label='Outer Loop Cmd')
            ax_vert.plot(time_values, list(actual_vert_vel_hist), 'c--', label='Follower Actual')
            ax_vert.set_title("Vertical Velocity (NED Down Positive)")
            ax_vert.set_xlabel("Time (s)")
            ax_vert.set_ylabel("Velocity (m/s)")
            ax_vert.legend(loc='upper left')

            plt.pause(0.01) # Yield to UI thread execution loop

        time.sleep(0.02) # Cap graph redraw attempts near ~50Hz

# Ignore roll/pitch/yaw body rates; target attitude quaternion & thrust only.
_ATTITUDE_ONLY_TYPE_MASK = 0b00000111

class CascadedFormationController:
    def __init__(self, leader_port, follower_port, offsets, visualizer_queue=None):
        self.vis_queue = visualizer_queue
        print(f"Connecting to Leader on {leader_port}...")
        self.leader = mavutil.mavlink_connection(leader_port)

        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)

        # Formation geometry offsets (Meters)
        self.f_c = offsets.get('f_c', 0.0) 
        self.l_c = offsets.get('l_c', 0.0) 
        self.v_c = offsets.get('v_c', 0.0) 

        # --- PHYSICAL AIRCRAFT PARAMETERS (Required for NLDI) ---
        # UPDATE THESE to match your JSBSim aircraft model
        self.AIRCRAFT_MASS = 5.9       # kg
        self.WING_AREA = 0.982           # m^2
        self.AIR_DENSITY = 1.225       # kg/m^3 (Sea level standard)
        self.CD_0 = 0.028               # Parasitic drag coefficient
        self.CD_ALPHA = 0.04           # Induced drag coefficient multiplier
        self.CL_0 = 0.25       # was 0.28 — corrected to exact table value at alpha=0
        self.CL_ALPHA = 5.0    # was 4.5  — corrected to slope of bracketing segment
        self.MAX_THRUST_NEWTONS = 38.9 # Max engine thrust in Newtons
        self.GRAVITY = 9.81            # m/s^2
        
        # --- OUTER LOOP: KINEMATIC GAINS ---
        # How aggressively to close the distance gaps (Position Error -> Target Velocity)
        self.K_P_POS_ALONG = 0.1
        self.K_D_POS_ALONG = 0.0  # <-- NEW: Along-track damping gain

        self.K_P_POS_CROSS = 0.15
        self.K_P_POS_VERT  = 0.9

        self.last_along_err = 0.0
        self.is_first_loop = True   # Prevents a massive derivative spike on startup

        self.K_TURN_FEEDFORWARD = 0.9
        
        # How aggressively to achieve target velocities (Velocity Error -> Target Acceleration)
        self.K_P_VEL = 0.6
        
        # Limiters
        self.MAX_ROLL = math.radians(45)   
        self.MAX_PITCH = math.radians(20)
        self.MAX_SPEED_DELTA = 5.0 # Max relative speed difference allowed (m/s)

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
            # REMOVE the chi_L calculation from here entirely

        elif msg.get_type() == 'ATTITUDE':
            self.leader_roll = msg.roll
            self.chi_L = msg.yaw  # Keep this as your single source of truth for orientation

    def _handle_follower_msg(self, msg):
        if msg.get_type() == 'LOCAL_POSITION_NED':
            self.follower_ned['n'], self.follower_ned['e'], self.follower_ned['d'] = msg.x, msg.y, msg.z
            self.follower_ned['vn'], self.follower_ned['ve'], self.follower_ned['vd'] = msg.vx, msg.vy, msg.vz
            self.follower_ned['valid'] = True
        
        elif msg.get_type() == 'ATTITUDE':
            self.follower_roll = msg.roll
            self.chi_F = msg.yaw  # Keep this as your single source of truth for orientation

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
                        # 1. OUTER LOOP (KINEMATICS & GEOMETRY)
                        # ==========================================
                        
                        # Target Slot Position
                        cos_c, sin_c = math.cos(self.chi_L), math.sin(self.chi_L)
                        slot_n = l_n - (self.f_c * cos_c + self.l_c * sin_c)
                        slot_e = l_e - (self.f_c * sin_c - self.l_c * cos_c)
                        slot_d = l_d + self.v_c

                        # Position Errors (Earth Frame)
                        err_n = slot_n - f_n
                        err_e = slot_e - f_e
                        err_d = slot_d - f_d

                        # Position Errors (Formation Body Frame)
                        along_err = err_n * cos_c + err_e * sin_c
                        cross_err = -err_n * sin_c + err_e * cos_c

                        if self.is_first_loop:
                            along_err_dot = 0
                            self.is_first_loop = False
                        else:
                            along_err_dot = (along_err - self.last_along_err) / self.dt
                            self.last_along_err = along_err

                        # Desired Velocities
                        l_speed = math.hypot(self.leader_ned['vn'], self.leader_ned['ve'])
                        f_speed = math.hypot(self.follower_ned['vn'], self.follower_ned['ve'])

                        # Estimate leader yaw rate from coordinated turn physics: omega = g * tan(roll) / V
                        turn_rate_leader = (self.GRAVITY / max(l_speed, 1.0)) * math.tan(self.leader_roll)

                        # Desired slot velocity = leader velocity + rotational velocity of the offset point.
                        # This gives the follower the velocity needed to stay rigidly attached during turns.
                        slot_vn = self.leader_ned['vn'] + turn_rate_leader * (self.f_c * sin_c - self.l_c * cos_c)
                        slot_ve = self.leader_ned['ve'] + turn_rate_leader * (-self.f_c * cos_c - self.l_c * sin_c)

                        slot_along_vel = slot_vn * cos_c + slot_ve * sin_c
                        slot_cross_vel = -slot_vn * sin_c + slot_ve * cos_c

                        # Apply proportional position feedback around the moving slot velocity.
                        target_along_vel = slot_along_vel + (along_err * self.K_P_POS_ALONG) + (along_err_dot * self.K_D_POS_ALONG)

                        # Constrain the target speed relative to the leader's actual speed
                        target_along_vel = max(l_speed - self.MAX_SPEED_DELTA, min(l_speed + self.MAX_SPEED_DELTA, target_along_vel))

                        target_cross_vel = slot_cross_vel + (cross_err * self.K_P_POS_CROSS)
                        target_vert_vel  = self.leader_ned['vd'] + (err_d * self.K_P_POS_VERT)

                        current_along_vel = self.follower_ned['vn'] * cos_c + self.follower_ned['ve'] * sin_c
                        current_cross_vel = -self.follower_ned['vn'] * sin_c + self.follower_ned['ve'] * cos_c

                        # Required Accelerations
                        accel_along = (target_along_vel - current_along_vel) * self.K_P_VEL
                        
                        # --- CROSS-TRACK ACCELERATION BREAKDOWN FOR DIAGNOSTICS ---
                        # 1. Proportional feedback component (based on position and velocity tracking errors)
                        accel_cross_feedback = (target_cross_vel - current_cross_vel) * self.K_P_VEL
                        
                        roll_blend = max(0, 1.0 - abs(self.leader_roll) / math.radians(45))
                        accel_cross_feedback *= roll_blend
                        
                        # 2. Feedforward component (based on leader's bank angle)
                        turn_rate_leader = (self.GRAVITY / max(l_speed, 1.0)) * math.tan(self.leader_roll)
                        feedforward_lat_accel = l_speed * turn_rate_leader
                        accel_cross_feedforward = self.K_TURN_FEEDFORWARD * feedforward_lat_accel
                        
                        # Total lateral acceleration requested
                        accel_cross = accel_cross_feedback + accel_cross_feedforward
                        
                        accel_vert = (target_vert_vel - self.follower_ned['vd']) * self.K_P_VEL

                        # ==========================================
                        # 2. INNER LOOP (NONLINEAR DYNAMIC INVERSION)
                        # ==========================================
                        
                        q_bar = 0.5 * self.AIR_DENSITY * (f_speed ** 2)
                        estimated_drag = q_bar * self.WING_AREA * self.CD_0 
                        
                        req_thrust_newtons = (self.AIRCRAFT_MASS * accel_along) + estimated_drag
                        throttle_cmd = max(0.0, min(1.0, req_thrust_newtons / self.MAX_THRUST_NEWTONS))
                        # Inside the run loop, smooth out the throttle command
    
                        
                        # Calculate final roll command
                        roll_cmd = math.atan(accel_cross / self.GRAVITY)
                        roll_cmd = max(-self.MAX_ROLL, min(self.MAX_ROLL, roll_cmd))

                        # Pitch and Yaw
                        lift_accel_req = self.GRAVITY - accel_vert 
                        cos_roll = max(math.cos(roll_cmd), 0.5)
                        pitch_cmd = math.atan((lift_accel_req / cos_roll - self.GRAVITY) / max(f_speed, 1.0))
                        pitch_cmd = max(-self.MAX_PITCH, min(self.MAX_PITCH, pitch_cmd))
                        target_yaw = self.chi_L + math.atan2(target_cross_vel, max(f_speed, 1.0))
                        
                        self.send_attitude_target(roll_cmd, pitch_cmd, 0, throttle_cmd)

                        if self.vis_queue is not None:
                            payload = {
                                'target_along_vel': target_along_vel,
                                'actual_along_vel': current_along_vel,
                                'target_cross_vel': target_cross_vel,
                                'actual_cross_vel': current_cross_vel,
                                'target_vert_vel': target_vert_vel,
                                'actual_vert_vel': self.follower_ned['vd'],
                                'roll_cmd': roll_cmd, 'throttle_cmd': throttle_cmd
                            }
                            # Non-blocking put: if visualizer slows down, ignore frame to avoid locking controller
                            try:
                                self.vis_queue.put_nowait(payload)
                            except multiprocessing.queues.Full:
                                pass

                        # --- ENHANCED DIAGNOSTIC TELEMETRY ---
                        if now - self._last_log_time >= self.log_interval_s:
                            self._last_log_time = now
                            print(f"Cross Err: {cross_err:+5.1f}m")
                            print(f"Follow Err: {along_err:+5.1f}m")
                            print(f"Alt Err: {err_d:+5.1f}m\n")

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("\nShutting down controller.")

if __name__ == "__main__":
    clearances = {'f_c': 5.0, 'l_c': 5.0, 'v_c': 0}

    shared_queue = multiprocessing.Queue(maxsize=10)
    
    # Spawn the independent visualizer process first
    plot_process = multiprocessing.Process(target=run_visualizer, args=(shared_queue,))
    plot_process.daemon = True
    plot_process.start()


    controller = CascadedFormationController("udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances, visualizer_queue=shared_queue)
    controller.run()