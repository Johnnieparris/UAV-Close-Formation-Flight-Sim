import time
import math
import numpy as np
from pymavlink import mavutil

_ATTITUDE_ONLY_TYPE_MASK = 0b00000111


# ---------------------------------------------------------------------------
# Lightweight Kinematic MPC
# ---------------------------------------------------------------------------
# State:  x = [along_pos, cross_pos, vert_pos]   (formation body frame)
# Input:  u = [v_along, v_cross, v_vert]          (desired velocity commands)
#
# Dynamics (Euler, dt):
#   x[k+1] = x[k] + dt * (u[k] - v_leader_body[k])
#
# Because the slot rotates with the leader's heading, we re-express the
# leader's NED velocity in formation-body frame at each prediction step,
# effectively giving us a linear, time-varying (but pre-computable) model.
#
# Cost:    J = sum_{k=1}^{N} x[k]'Qx[k] + sum_{k=0}^{N-1} u[k]'Ru[k]
#              + x[N]'P_f x[N]
#
# Solved analytically via batch least-squares (O(N) build, O(n^3) solve once
# per loop with n = 3N — cheap at N <= 20 and 3 states).
# ---------------------------------------------------------------------------

class KinematicMPC:
    """
    Receding-horizon kinematic MPC for one follower UAV.

    Parameters
    ----------
    N       : prediction horizon (steps)
    dt      : sample time (s) — must match outer-loop rate
    Q       : 3x3 state cost matrix (position tracking)
    R       : 3x3 input cost matrix (velocity effort)
    P_f     : 3x3 terminal state cost (>= Q recommended for stability)
    v_max   : maximum allowed speed command per axis (m/s)
    """

    def __init__(self, N=10, dt=0.05, Q=None, R=None, P_f=None, v_max=8.0):
        self.N    = N
        self.dt   = dt
        self.v_max = v_max

        # Default cost matrices (diagonal, tunable)
        self.Q   = Q   if Q   is not None else np.diag([1.2, 2.0, 3.0])
        self.R   = R   if R   is not None else np.diag([0.3, 0.4, 0.2])
        self.P_f = P_f if P_f is not None else np.diag([3.0, 5.0, 6.0])

    def _build_batch_matrices(self, leader_body_velocities):
        """
        Build the batch prediction matrices Phi and Gamma such that
            X = Phi * x0 + Gamma * U
        where X stacks [x1, x2, ..., xN] and U stacks [u0, ..., u_{N-1}].

        leader_body_velocities : list of N np.array([vl_along, vl_cross, vl_vert])
        """
        n, m, N, dt = 3, 3, self.N, self.dt

        # Phi maps initial state to free response
        Phi = np.zeros((N * n, n))
        # Gamma maps inputs to constrained response
        Gamma = np.zeros((N * n, N * m))

        A = np.eye(n)  # State transition (identity — pure integrator kinematics)

        A_pow = np.eye(n)
        for k in range(N):
            A_pow = A_pow @ A  # stays identity here; kept for easy extension
            Phi[k*n:(k+1)*n, :] = A_pow

        for k in range(N):
            # Influence of u_j on x_{k+1}
            for j in range(k + 1):
                step_power = k - j
                A_pow_ij = np.eye(n)  # A^(step_power) — identity for integrator
                B_eff = dt * A_pow_ij  # B = dt*I, already in body frame
                Gamma[k*n:(k+1)*n, j*m:(j+1)*m] += B_eff

        # Disturbance offset: leader velocity integrated forward creates a
        # "drift" that the MPC must counteract. We fold it into the reference.
        # d[k] = -dt * v_leader_body[k]  (slot moves with leader)
        D = np.zeros(N * n)
        A_pow = np.eye(n)
        for k in range(N):
            # Accumulate disturbance from step j onto state k+1
            for j in range(k + 1):
                step = k - j
                # Each disturbance d[j] propagates A^step steps
                D[k*n:(k+1)*n] += (-dt) * leader_body_velocities[j]

        return Phi, Gamma, D

    def _build_cost_matrices(self):
        """Stack Q, R, P_f into block-diagonal matrices for the batch problem."""
        n, m, N = 3, 3, self.N
        Q_bar = np.zeros((N * n, N * n))
        R_bar = np.zeros((N * m, N * m))

        for k in range(N - 1):
            Q_bar[k*n:(k+1)*n, k*n:(k+1)*n] = self.Q
        # Terminal cost on last state
        Q_bar[(N-1)*n:N*n, (N-1)*n:N*n] = self.P_f

        for k in range(N):
            R_bar[k*m:(k+1)*m, k*m:(k+1)*m] = self.R

        return Q_bar, R_bar

    def compute(self, x0, leader_body_velocities):
        """
        Solve one MPC step.

        Parameters
        ----------
        x0                    : np.array([along_err, cross_err, vert_err])
                                Current position error in formation body frame.
        leader_body_velocities: list of N np.array([vl_along, vl_cross, vl_vert])
                                Leader velocity projected into formation body frame
                                for each prediction step (can repeat current if
                                leader heading is assumed constant).

        Returns
        -------
        u_opt : np.array([v_along_cmd, v_cross_cmd, v_vert_cmd])
                First optimal velocity command to apply (receding horizon).
        predicted_errors : list of N np.array  (for diagnostics)
        """
        N, n, m = self.N, 3, 3

        Phi, Gamma, D = self._build_batch_matrices(leader_body_velocities)
        Q_bar, R_bar  = self._build_cost_matrices()

        # Free response (what happens with U=0 due to disturbance)
        X_free = Phi @ x0 + D

        # Unconstrained QP: dJ/dU = 0
        # J = (Gamma U + X_free)' Q_bar (Gamma U + X_free) + U' R_bar U
        # => (Gamma' Q_bar Gamma + R_bar) U = -Gamma' Q_bar X_free
        H = Gamma.T @ Q_bar @ Gamma + R_bar
        g = Gamma.T @ Q_bar @ X_free

        # Solve via Cholesky (H is symmetric positive definite)
        try:
            U_opt = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Fallback to least-squares if H is ill-conditioned
            U_opt, _, _, _ = np.linalg.lstsq(H, -g, rcond=None)

        # Extract first control action (receding horizon principle)
        u_opt = U_opt[:m]

        # Clamp to velocity limits
        u_opt = np.clip(u_opt, -self.v_max, self.v_max)

        # Predicted state trajectory (for diagnostics)
        X_pred = Gamma @ U_opt + X_free
        predicted_errors = [X_pred[k*n:(k+1)*n] for k in range(N)]

        return u_opt, predicted_errors

    def update_dt(self, dt):
        """Update sample time (call if dt varies significantly)."""
        self.dt = max(0.001, dt)


# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------

class CascadedFormationController:
    def __init__(self, leader_port, follower_port, offsets):
        print(f"Connecting to Leader on {leader_port}...")
        self.leader = mavutil.mavlink_connection(leader_port)

        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)

        # Formation geometry offsets (meters, formation body frame)
        # f_c = forward clearance, l_c = lateral clearance, v_c = vertical
        self.f_c = offsets.get('f_c', 0.0)
        self.l_c = offsets.get('l_c', 0.0)
        self.v_c = offsets.get('v_c', 0.0)

        # --- PHYSICAL AIRCRAFT PARAMETERS ---
        self.AIRCRAFT_MASS        = 5.9
        self.WING_AREA            = 0.982
        self.AIR_DENSITY          = 1.225
        self.CD_0                 = 0.028
        self.CD_ALPHA             = 0.04
        self.CL_0                 = 0.25
        self.CL_ALPHA             = 5.0
        self.MAX_THRUST_NEWTONS   = 38.9
        self.GRAVITY              = 9.81

        # --- MPC OUTER LOOP ---
        # Horizon: 10 steps at 50ms = 0.5s look-ahead.
        # Increase N (e.g. 20) for tighter formation in sustained turns at the
        # cost of ~4x solve time (still <1ms on any modern CPU).
        #
        # Cost weights:
        #   Q  — penalises position error [along, cross, vert]
        #        Higher cross/vert keeps tight lateral & altitude formation.
        #   R  — penalises velocity effort (smooth commands, lower = more
        #        aggressive corrections).
        #   P_f — terminal weight; set >= Q for recursive feasibility.
        self.mpc = KinematicMPC(
            N    = 10,
            dt   = 0.05,
            Q    = np.diag([1.2, 2.5, 4.0]),   # [along, cross, vert]
            R    = np.diag([0.4, 0.5, 0.25]),
            P_f  = np.diag([3.0, 6.0, 8.0]),
            v_max= 8.0,
        )

        # Feedforward turn gain (blended into MPC via leader vel prediction)
        self.K_TURN_FEEDFORWARD = 0.9

        # --- INNER LOOP: NLDI GAINS (unchanged from original) ---
        self.K_P_VEL = 0.6

        # Limiters
        self.MAX_ROLL        = math.radians(45)
        self.MAX_PITCH       = math.radians(20)
        self.MAX_SPEED_DELTA = 5.0

        # State
        self.leader_ned   = self._empty_ned()
        self.follower_ned = self._empty_ned()
        self.leader_roll  = 0.0
        self.has_offset   = False
        self.offset       = {'n': 0.0, 'e': 0.0, 'd': 0.0}
        self.chi_L        = 0.0
        self.chi_F        = 0.0
        self.dt           = 0.05

        self._last_log_time = 0.0
        self.log_interval_s = 0.5

        # Leader heading rate estimate (for MPC prediction of future chi_L)
        self._prev_chi_L     = 0.0
        self._chi_dot_L      = 0.0   # rad/s — low-pass filtered
        self._CHI_DOT_ALPHA  = 0.15  # LP filter coefficient (lower = smoother)

        self.leader.wait_heartbeat()
        print("Leader heartbeat verified.")
        self.follower.wait_heartbeat()
        print("Follower heartbeat verified.")

        self._request_message_stream(
            self.leader, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)
        self._request_message_stream(
            self.leader, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50_000)
        self._request_message_stream(
            self.follower, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 100_000)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_ned():
        return {'n': 0.0, 'e': 0.0, 'd': 0.0,
                'vn': 0.0, 've': 0.0, 'vd': 0.0, 'valid': False}

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
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
                mode_id,
            )

    def _drain_messages(self, link, handler):
        while True:
            msg = link.recv_msg()
            if msg is None:
                break
            handler(msg)

    def _handle_leader_msg(self, msg):
        t = msg.get_type()
        if t == 'LOCAL_POSITION_NED':
            self.leader_ned.update({
                'n': msg.x, 'e': msg.y, 'd': msg.z,
                'vn': msg.vx, 've': msg.vy, 'vd': msg.vz, 'valid': True,
            })
            speed = math.hypot(msg.vx, msg.vy)
            if speed > 0.5:
                new_chi = math.atan2(msg.vy, msg.vx)
                # Low-pass filter heading rate
                raw_dot = (new_chi - self._prev_chi_L) / max(self.dt, 0.001)
                # Wrap to [-pi, pi]
                raw_dot = (raw_dot + math.pi) % (2 * math.pi) - math.pi
                self._chi_dot_L = ((1 - self._CHI_DOT_ALPHA) * self._chi_dot_L
                                   + self._CHI_DOT_ALPHA * raw_dot)
                self._prev_chi_L = new_chi
                self.chi_L = new_chi
        elif t == 'ATTITUDE':
            self.leader_roll = msg.roll
            self.chi_L       = msg.yaw
            # Derive heading rate from physics when roll is available
            l_speed = math.hypot(self.leader_ned['vn'], self.leader_ned['ve'])
            if l_speed > 1.0:
                omega_physics = (self.GRAVITY / l_speed) * math.tan(self.leader_roll)
                self._chi_dot_L = ((1 - self._CHI_DOT_ALPHA) * self._chi_dot_L
                                   + self._CHI_DOT_ALPHA * omega_physics)

    def _handle_follower_msg(self, msg):
        if msg.get_type() == 'LOCAL_POSITION_NED':
            self.follower_ned.update({
                'n': msg.x, 'e': msg.y, 'd': msg.z,
                'vn': msg.vx, 've': msg.vy, 'vd': msg.vz, 'valid': True,
            })
            speed = math.hypot(msg.vx, msg.vy)
            if speed > 0.5:
                self.chi_F = math.atan2(msg.vy, msg.vx)

    def _sync_offset(self):
        self.offset  = {'n': 0.0, 'e': 0.0, 'd': 0.0}
        self.has_offset = True

    def send_attitude_target(self, roll, pitch, yaw, thrust):
        q = self.euler_to_quaternion(roll, pitch, yaw)
        self.follower.mav.set_attitude_target_send(
            0, self.follower.target_system, self.follower.target_component,
            _ATTITUDE_ONLY_TYPE_MASK, q, 0, 0, 0, thrust,
        )

    # ------------------------------------------------------------------
    # MPC leader velocity prediction
    # ------------------------------------------------------------------

    def _predict_leader_body_velocities(self, l_speed, chi_L_now):
        """
        Build N predicted leader velocity vectors in the *current* formation
        body frame.  We propagate chi_L forward using the estimated heading
        rate and compute the speed-direction decomposition at each step.

        The lateral slot offset (l_c) modifies the effective along-track speed
        via the coordinated-turn radius relationship, exactly as in the original
        wingman compensation block.
        """
        preds = []
        chi = chi_L_now
        chi_dot = self._chi_dot_L
        dt = self.dt

        for k in range(self.mpc.N):
            # Predicted heading k steps ahead
            chi_k = chi + chi_dot * k * dt

            # In formation body frame (aligned with chi_L_now at step 0),
            # leader's along/cross velocities:
            delta_chi = chi_k - chi_L_now
            vl_along =  l_speed * math.cos(delta_chi)
            vl_cross =  l_speed * math.sin(delta_chi)

            # Radius compensation: inner/outer wingman speed correction
            # omega * l_c gives the lateral speed penalty/bonus
            omega = chi_dot
            vl_along_adj = vl_along - omega * self.l_c

            preds.append(np.array([vl_along_adj, vl_cross, self.leader_ned['vd']]))

        return preds

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("Syncing frames...")
        while not (self.has_offset
                   and self.leader_ned['valid']
                   and self.follower_ned['valid']):
            self._drain_messages(self.leader,   self._handle_leader_msg)
            self._drain_messages(self.follower, self._handle_follower_msg)
            if (not self.has_offset
                    and self.leader_ned['valid']
                    and self.follower_ned['valid']):
                self._sync_offset()
            time.sleep(0.05)

        self.set_follower_mode('GUIDED')
        print("Running MPC Outer Loop + NLDI Inner Loop\n")

        last_time = time.monotonic()

        try:
            while True:
                now      = time.monotonic()
                self.dt  = max(0.001, now - last_time)
                last_time = now
                self.mpc.update_dt(self.dt)

                self._drain_messages(self.leader,   self._handle_leader_msg)
                self._drain_messages(self.follower, self._handle_follower_msg)

                if not self.has_offset:
                    time.sleep(0.05)
                    continue

                # ---- Current states ----
                l_n, l_e, l_d = (self.leader_ned['n'],
                                  self.leader_ned['e'],
                                  self.leader_ned['d'])
                l_speed = math.hypot(self.leader_ned['vn'],
                                     self.leader_ned['ve'])
                f_speed = math.hypot(self.follower_ned['vn'],
                                     self.follower_ned['ve'])

                f_n = self.follower_ned['n'] + self.offset['n']
                f_e = self.follower_ned['e'] + self.offset['e']
                f_d = self.follower_ned['d'] + self.offset['d']

                # =====================================================
                # 1.  OUTER LOOP — MPC (replaces P-control)
                # =====================================================

                cos_c = math.cos(self.chi_L)
                sin_c = math.sin(self.chi_L)

                # Desired slot position (NED)
                slot_n = l_n - (self.f_c * cos_c + self.l_c * sin_c)
                slot_e = l_e - (self.f_c * sin_c - self.l_c * cos_c)
                slot_d = l_d + self.v_c

                # Position errors in NED → formation body frame
                err_n = slot_n - f_n
                err_e = slot_e - f_e
                err_d = slot_d - f_d

                along_err = err_n * cos_c + err_e * sin_c
                cross_err = -err_n * sin_c + err_e * cos_c
                vert_err  = err_d

                x0 = np.array([along_err, cross_err, vert_err])

                # Predict leader's body-frame velocities over the horizon
                leader_body_vels = self._predict_leader_body_velocities(
                    l_speed, self.chi_L)

                # Solve MPC — returns optimal velocity command for this step
                u_opt, pred_errors = self.mpc.compute(x0, leader_body_vels)

                # u_opt = [v_along_cmd, v_cross_cmd, v_vert_cmd] in body frame
                target_along_vel = u_opt[0]
                target_cross_vel = u_opt[1]
                target_vert_vel  = u_opt[2]

                # Hard speed clamp relative to leader (safety envelope)
                target_along_vel = max(l_speed - self.MAX_SPEED_DELTA,
                                       min(l_speed + self.MAX_SPEED_DELTA,
                                           target_along_vel))

                # Velocity errors → accelerations (inner-loop input)
                accel_along = (target_along_vel - f_speed) * self.K_P_VEL

                # Cross: MPC already handles turn feedforward via prediction,
                # but we keep the roll-angle feedforward for large bank angles
                # (MPC horizon is finite; sharp inputs need direct feedthrough).
                current_cross_vel = (-self.follower_ned['vn'] * sin_c
                                     + self.follower_ned['ve'] * cos_c)
                accel_cross_feedback = ((target_cross_vel - current_cross_vel)
                                        * self.K_P_VEL)

                if abs(self.leader_roll) > math.radians(5):
                    accel_cross_feedback = 0.0

                feedforward_lat    = self.chi_L * l_speed
                accel_cross        = (accel_cross_feedback
                                      + self.K_TURN_FEEDFORWARD * feedforward_lat)

                current_vert_vel   = self.follower_ned['vd']
                accel_vert         = ((target_vert_vel - current_vert_vel)
                                      * self.K_P_VEL)

                # =====================================================
                # 2.  INNER LOOP — NLDI (unchanged)
                # =====================================================

                q_bar = 0.5 * self.AIR_DENSITY * (f_speed ** 2)
                estimated_drag = q_bar * self.WING_AREA * self.CD_0

                req_thrust = (self.AIRCRAFT_MASS * accel_along) + estimated_drag
                throttle_cmd = max(0.0, min(1.0,
                                            req_thrust / self.MAX_THRUST_NEWTONS))

                roll_cmd = math.atan(accel_cross / self.GRAVITY)
                roll_cmd = max(-self.MAX_ROLL, min(self.MAX_ROLL, roll_cmd))

                lift_accel_req = self.GRAVITY - accel_vert
                cos_roll       = max(math.cos(roll_cmd), 0.5)
                pitch_cmd      = math.atan(
                    (lift_accel_req / cos_roll - self.GRAVITY)
                    / max(f_speed, 1.0))
                pitch_cmd = max(-self.MAX_PITCH, min(self.MAX_PITCH, pitch_cmd))

                self.send_attitude_target(roll_cmd, pitch_cmd, 0, throttle_cmd)

                # ---- Diagnostics ----
                if now - self._last_log_time >= self.log_interval_s:
                    self._last_log_time = now
                    # Show predicted error at horizon end for MPC health check
                    pred_along = pred_errors[-1][0] if pred_errors else 0.0
                    pred_cross = pred_errors[-1][1] if pred_errors else 0.0
                    print(f"[MPC] Cross Err: {cross_err:+5.1f}m  "
                          f"Along Err: {along_err:+5.1f}m  "
                          f"Alt Err: {vert_err:+5.1f}m")
                    print(f"      Pred@N:   cross={pred_cross:+5.2f}m  "
                          f"along={pred_along:+5.2f}m  "
                          f"chi_dot={math.degrees(self._chi_dot_L):+5.2f}°/s\n")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nShutting down controller.")


if __name__ == "__main__":
    clearances = {'f_c': 5.0, 'l_c': 5.0, 'v_c': 0}
    controller = CascadedFormationController(
        "udp:127.0.0.1:14552", "udp:127.0.0.1:14562", clearances)
    controller.run()