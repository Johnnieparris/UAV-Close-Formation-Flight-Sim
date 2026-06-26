import time
import math
import numpy as np
from scipy.optimize import minimize
from pymavlink import mavutil

import multiprocessing
import matplotlib.pyplot as plt
from collections import deque

def run_visualizer(data_queue):
    """Runs in a separate process to handle real-time plotting."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_spd, ax_roll, ax_vert = axes
    fig.suptitle("MPC Outer Loop Commands vs Follower Response", fontsize=14, color='white')

    max_history = 100
    time_hist = deque(maxlen=max_history)
    
    target_spd_hist = deque(maxlen=max_history)
    actual_spd_hist = deque(maxlen=max_history)
    
    target_roll_hist = deque(maxlen=max_history)
    actual_roll_hist = deque(maxlen=max_history)
    
    target_vert_hist = deque(maxlen=max_history)
    actual_vert_hist = deque(maxlen=max_history)

    t_start = time.time()

    while plt.fignum_exists(fig.number):
        data = None
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
            except Exception:
                break

        if data is not None:
            time_hist.append(time.time() - t_start)
            
            target_spd_hist.append(data['target_speed'])
            actual_spd_hist.append(data['actual_speed'])
            
            target_roll_hist.append(math.degrees(data['target_roll']))
            actual_roll_hist.append(math.degrees(data['actual_roll']))
            
            target_vert_hist.append(data['target_vert_vel'])
            actual_vert_hist.append(data['actual_vert_vel'])

            time_values = list(time_hist)

            for ax in axes:
                ax.clear()
                ax.grid(True, color='gray', alpha=0.3)

            ax_spd.plot(time_values, list(target_spd_hist), 'y-', label='MPC Cmd Speed ($v^c$)')
            ax_spd.plot(time_values, list(actual_spd_hist), 'c--', label='Follower Actual ($v$)')
            ax_spd.set_title("Speed Tracking")
            ax_spd.set_ylabel("Speed (m/s)")
            ax_spd.legend(loc='upper left')

            ax_roll.plot(time_values, list(target_roll_hist), 'y-', label='MPC Cmd Roll ($\gamma^c$)')
            ax_roll.plot(time_values, list(actual_roll_hist), 'c--', label='Follower Actual ($\gamma$)')
            ax_roll.set_title("Roll Tracking (Coordinated Turn)")
            ax_roll.set_ylabel("Bank Angle (deg)")
            ax_roll.legend(loc='upper left')

            ax_vert.plot(time_values, list(target_vert_hist), 'y-', label='P-Loop Cmd Vel')
            ax_vert.plot(time_values, list(actual_vert_hist), 'c--', label='Follower Actual Vel')
            ax_vert.set_title("Vertical Velocity Tracking (Decoupled)")
            ax_vert.set_xlabel("Time (s)")
            ax_vert.set_ylabel("Velocity (m/s)")
            ax_vert.legend(loc='upper left')

            plt.pause(0.01) 

        time.sleep(0.02) 

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

        # --- PHYSICAL & INNER LOOP PARAMETERS ---
        self.AIRCRAFT_MASS = 5.9       
        self.WING_AREA = 0.982           
        self.AIR_DENSITY = 1.225       
        self.CD_0 = 0.028               
        self.MAX_THRUST_NEWTONS = 38.9 
        self.GRAVITY = 9.81            
        
        self.K_P_POS_VERT  = 0.9
        self.K_P_VEL = 0.6
        self.MAX_PITCH = math.radians(20)

        # --- MPC PARAMETERS (paper-aligned outer loop) ---
        self.N_horizon = 8
        self.dt_mpc = 0.5
        self.a_v = 3.2
        self.a_gamma = 0.6

        self.Q_pos = 1.0
        self.Q_speed = 0.05
        self.Q_heading = 0.5
        self.RHO_SPEED = 5
        self.RHO_ROLL = 500

        self.MAX_SPEED = 30.0
        self.MIN_SPEED = 10.0
        self.MAX_SPEED_RATE = 5.0
        self.MAX_ROLL_MPC = math.radians(30)

        # Optimizer state (warm start)
        self.last_commanded_speed = 0.5 * (self.MIN_SPEED + self.MAX_SPEED)
        self.last_optimal_u = np.tile(
            np.array([self.last_commanded_speed, 0.0]),
            self.N_horizon
        )

        # State tracking
        self.leader_ned = self._empty_ned()
        self.follower_ned = self._empty_ned()
        self.leader_roll = 0.0  
        self.follower_roll = 0.0
        self.has_offset = False
        self.offset = {'n': 0.0, 'e': 0.0, 'd': 0.0}
        self.chi_L = 0.0  
        self.chi_F = 0.0  
        self.dt = 0.05

        self.leader.wait_heartbeat()
        self.follower.wait_heartbeat()
        
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(self.leader, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000) 
        self._request_message_stream(self.follower, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(self.follower, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000) 

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

    # ==========================================
    # MPC OPTIMIZATION CORE
    # ==========================================
    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _roll_from_yaw_rate(self, speed, yaw_rate):
        roll = math.atan2(speed * yaw_rate, self.GRAVITY)
        return max(-self.MAX_ROLL_MPC, min(self.MAX_ROLL_MPC, roll))

    def build_reference_horizon(self, leader_speed, leader_turn_rate):
        """Predict the desired virtual formation point over the MPC horizon."""
        ref_traj = []
        pred_ln = self.leader_ned['n']
        pred_le = self.leader_ned['e']
        pred_chi = self.chi_L

        for _ in range(self.N_horizon):
            pred_ln += leader_speed * math.cos(pred_chi) * self.dt_mpc
            pred_le += leader_speed * math.sin(pred_chi) * self.dt_mpc
            pred_chi = self._wrap_angle(pred_chi + leader_turn_rate * self.dt_mpc)

            cos_c, sin_c = math.cos(pred_chi), math.sin(pred_chi)
            slot_n = pred_ln - (self.f_c * cos_c + self.l_c * sin_c)
            slot_e = pred_le - (self.f_c * sin_c - self.l_c * cos_c)

            slot_vn = (
                leader_speed * cos_c
                + leader_turn_rate * (self.f_c * sin_c - self.l_c * cos_c)
            )
            slot_ve = (
                leader_speed * sin_c
                + leader_turn_rate * (-self.f_c * cos_c - self.l_c * sin_c)
            )
            slot_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, math.hypot(slot_vn, slot_ve)))
            slot_heading = math.atan2(slot_ve, slot_vn)
            ref_roll = self._roll_from_yaw_rate(slot_speed, leader_turn_rate)

            ref_traj.append({
                'n': slot_n,
                'e': slot_e,
                'speed': slot_speed,
                'heading': slot_heading,
                'roll': ref_roll,
            })

        return ref_traj

    def _shift_warm_start(self, control_sequence):
        shifted = np.empty_like(control_sequence)
        shifted[:-2] = control_sequence[2:]
        shifted[-2:] = control_sequence[-2:]
        return shifted

    def mpc_cost_function(self, U, x0, reference_traj):
        """
        Evaluates the cost of a control sequence U = [v_c_0, gamma_c_0, v_c_1, gamma_c_1, ...]
        x0 = [n, e, psi, v, gamma]
        """
        cost = 0.0
        n, e, psi, v, gamma = x0
        
        for k in range(self.N_horizon):
            v_c = U[2*k]
            gamma_c = U[2*k + 1]
            
            # 1. Forward Kinematic Model (from paper equations)
            # n = x (North), e = y (East) in standard MAVLink
            dn = v * math.cos(psi)
            de = v * math.sin(psi)
            dpsi = (self.GRAVITY * math.tan(gamma)) / max(v, 1.0)
            dv = (1.0 / self.a_v) * (v_c - v)
            dgamma = (1.0 / self.a_gamma) * (gamma_c - gamma)
            
            # Euler integration
            n += dn * self.dt_mpc
            e += de * self.dt_mpc
            psi = self._wrap_angle(psi + dpsi * self.dt_mpc)
            v += dv * self.dt_mpc
            gamma += dgamma * self.dt_mpc

            ref = reference_traj[k]
            pos_error_sq = (n - ref['n'])**2 + (e - ref['e'])**2
            speed_error_sq = (v - ref['speed'])**2
            heading_error_sq = self._wrap_angle(psi - ref['heading'])**2
            control_speed_error_sq = (v_c - ref['speed'])**2
            control_roll_error_sq = (gamma_c - ref['roll'])**2

            cost += self.Q_pos * pos_error_sq
            cost += self.Q_speed * speed_error_sq
            cost += self.Q_heading * heading_error_sq
            cost += self.RHO_SPEED * control_speed_error_sq
            cost += self.RHO_ROLL * control_roll_error_sq

        return cost

    def run(self):
            print("Syncing frames...")
            while not (self.has_offset and self.leader_ned['valid'] and self.follower_ned['valid']):
                self._drain_messages(self.leader, self._handle_leader_msg)
                self._drain_messages(self.follower, self._handle_follower_msg)
                if not self.has_offset and self.leader_ned['valid'] and self.follower_ned['valid']:
                    self._sync_offset()
                time.sleep(0.05)
                
            self.set_follower_mode('GUIDED')
            print("Running Cascaded Controller: MPC Outer Loop + NLDI Inner Loop\n")

            last_time = time.monotonic()

            try:
                while True:
                    now = time.monotonic()
                    self.dt = max(0.001, now - last_time)
                    last_time = now

                    self._drain_messages(self.leader, self._handle_leader_msg)
                    self._drain_messages(self.follower, self._handle_follower_msg)

                    if self.has_offset:
                        # ------------------------------------------
                        # 1. GENERATE PAPER-STYLE REFERENCE HORIZON
                        # ------------------------------------------
                        l_speed = math.hypot(self.leader_ned['vn'], self.leader_ned['ve'])
                        turn_rate_leader = (self.GRAVITY / max(l_speed, 1.0)) * math.tan(self.leader_roll)
                        reference_traj = self.build_reference_horizon(l_speed, turn_rate_leader)

                        # ------------------------------------------
                        # 2. RUN PAPER-ALIGNED KINEMATIC MPC
                        # ------------------------------------------
                        f_speed = math.hypot(self.follower_ned['vn'], self.follower_ned['ve'])
                        x0 = [
                            self.follower_ned['n'] + self.offset['n'], 
                            self.follower_ned['e'] + self.offset['e'], 
                            self.chi_F, 
                            f_speed, 
                            self.follower_roll
                        ]
                        
                        # Bounds for the optimizer: [Speed, Roll] for each step.
                        # The paper constrains speed rate; this first-step bound
                        # prevents the applied command from jumping rail-to-rail.
                        max_speed_step = self.MAX_SPEED_RATE * self.dt_mpc
                        first_min_speed = max(self.MIN_SPEED, self.last_commanded_speed - max_speed_step)
                        first_max_speed = min(self.MAX_SPEED, self.last_commanded_speed + max_speed_step)
                        bnds = []
                        for k in range(self.N_horizon):
                            if k == 0:
                                bnds.append((first_min_speed, first_max_speed))
                            else:
                                bnds.append((self.MIN_SPEED, self.MAX_SPEED))
                            bnds.append((-self.MAX_ROLL_MPC, self.MAX_ROLL_MPC))

                        initial_guess = self._shift_warm_start(self.last_optimal_u)
                        initial_guess[0] = max(first_min_speed, min(first_max_speed, initial_guess[0]))

                        # Optimize over commanded speed and commanded roll.
                        res = minimize(
                            self.mpc_cost_function, 
                            initial_guess, 
                            args=(x0, reference_traj), 
                            bounds=bnds,
                            method='SLSQP',
                            options={'maxiter': 100, 'ftol': 1e-3}
                        )

                        if res.success:
                            self.last_optimal_u = res.x
                        else:
                            self.last_optimal_u = initial_guess
                        
                        # Extract the first optimal command
                        target_speed = self.last_optimal_u[0]
                        target_roll = self.last_optimal_u[1]
                        self.last_commanded_speed = target_speed

                        # ------------------------------------------
                        # 3. VERTICAL DECOUPLED LOOP
                        # ------------------------------------------
                        target_slot_d = self.leader_ned['d'] + self.v_c
                        err_d = target_slot_d - (self.follower_ned['d'] + self.offset['d'])
                        target_vert_vel = self.leader_ned['vd'] + (err_d * self.K_P_POS_VERT)

                        # ------------------------------------------
                        # 4. NLDI INNER LOOP (Convert targets to forces)
                        # ------------------------------------------
                        # Along-track acceleration required to hit target speed
                        accel_along = (target_speed - f_speed) * self.K_P_VEL
                        
                        q_bar = 0.5 * self.AIR_DENSITY * (max(f_speed, 1.0) ** 2)
                        estimated_drag = q_bar * self.WING_AREA * self.CD_0 
                        
                        req_thrust_newtons = (self.AIRCRAFT_MASS * accel_along) + estimated_drag
                        throttle_cmd = max(0.0, min(1.0, req_thrust_newtons / self.MAX_THRUST_NEWTONS))
                        
                        # Vertical acceleration required
                        accel_vert = (target_vert_vel - self.follower_ned['vd']) * self.K_P_VEL
                        lift_accel_req = self.GRAVITY - accel_vert 
                        
                        cos_roll = max(math.cos(target_roll), 0.5)
                        pitch_cmd = math.atan((lift_accel_req / cos_roll - self.GRAVITY) / max(f_speed, 1.0))
                        pitch_cmd = max(-self.MAX_PITCH, min(self.MAX_PITCH, pitch_cmd))
                        
                        # Send MPC-derived roll and NLDI-derived pitch/throttle
                        self.send_attitude_target(target_roll, pitch_cmd, 0, throttle_cmd)

                        if self.vis_queue is not None:
                            payload = {
                                'target_speed': target_speed,
                                'actual_speed': f_speed,
                                'target_roll': target_roll,
                                'actual_roll': self.follower_roll,
                                'target_vert_vel': target_vert_vel,
                                'actual_vert_vel': self.follower_ned['vd']
                            }
                            try:
                                self.vis_queue.put_nowait(payload)
                            except multiprocessing.queues.Full:
                                pass

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("\nShutting down controller.")

if __name__ == "__main__":
    clearances = {'f_c': 5.0, 'l_c': 5.0, 'v_c': 0}

    shared_queue = multiprocessing.Queue(maxsize=10)
    
    plot_process = multiprocessing.Process(target=run_visualizer, args=(shared_queue,))
    plot_process.daemon = True
    plot_process.start()

    controller = CascadedFormationController("udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances, visualizer_queue=shared_queue)
    controller.run()