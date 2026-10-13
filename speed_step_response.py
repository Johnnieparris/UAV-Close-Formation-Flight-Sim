import argparse
import csv
import math
import time

from pymavlink import mavutil


# Ignore roll/pitch/yaw body rates; target attitude quaternion & thrust only.
_ATTITUDE_ONLY_TYPE_MASK = 0b00000111


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


class FollowerSpeedStepResponse:
    def __init__(self, follower_port):
        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.vn = 0.0
        self.ve = 0.0
        self.speed = 0.0
        self.has_attitude = False
        self.has_position = False

        # Same values used by MPC-NLDI.py for the follower speed inner loop.
        self.aircraft_mass = 5.9
        self.wing_area = 0.982
        self.air_density = 1.225
        self.cd_0 = 0.028
        self.max_thrust_newtons = 38.9
        self.k_p_vel = 0.6

        self.follower.wait_heartbeat()
        print("Follower heartbeat verified.")
        self._request_message_stream(
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            20_000,
        )
        self._request_message_stream(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            20_000,
        )

    def _request_message_stream(self, msg_id, interval_us):
        self.follower.mav.command_long_send(
            self.follower.target_system,
            self.follower.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def set_follower_mode(self, mode_name):
        mode_mapping = self.follower.mode_mapping()
        mode_id = mode_mapping.get(mode_name)
        if mode_id is None:
            raise RuntimeError(f"Unknown follower mode: {mode_name}")

        self.follower.mav.set_mode_send(
            self.follower.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

    def _drain_messages(self):
        updated = False
        while True:
            msg = self.follower.recv_msg()
            if msg is None:
                break

            msg_type = msg.get_type()
            if msg_type == "ATTITUDE":
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
                self.has_attitude = True
                updated = True
            elif msg_type == "LOCAL_POSITION_NED":
                self.vn = msg.vx
                self.ve = msg.vy
                self.speed = math.hypot(self.vn, self.ve)
                self.has_position = True
                updated = True

        return updated

    def wait_for_state(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._drain_messages()
            if self.has_attitude and self.has_position:
                return
            time.sleep(0.02)
        raise TimeoutError("Timed out waiting for follower attitude and speed.")

    def send_attitude_target(self, roll, pitch, yaw, thrust):
        q = euler_to_quaternion(roll, pitch, yaw)
        self.follower.mav.set_attitude_target_send(
            0,
            self.follower.target_system,
            self.follower.target_component,
            _ATTITUDE_ONLY_TYPE_MASK,
            q,
            0,
            0,
            0,
            thrust,
        )

    def speed_to_throttle(self, target_speed):
        accel_along = (target_speed - self.speed) * self.k_p_vel
        q_bar = 0.5 * self.air_density * (max(self.speed, 1.0) ** 2)
        estimated_drag = q_bar * self.wing_area * self.cd_0
        req_thrust_newtons = (self.aircraft_mass * accel_along) + estimated_drag
        return max(0.0, min(1.0, req_thrust_newtons / self.max_thrust_newtons))

    @staticmethod
    def reached_threshold(actual_speed, initial_speed, command_speed, threshold_speed):
        if command_speed >= initial_speed:
            return actual_speed >= threshold_speed
        return actual_speed <= threshold_speed

    def run(
        self,
        command_speed,
        response_fraction,
        duration_s,
        rate_hz,
        settle_s,
        csv_path,
    ):
        sample_period = 1.0 / rate_hz

        self.wait_for_state(timeout_s=5.0)
        self.set_follower_mode("GUIDED")
        time.sleep(settle_s)
        self.wait_for_state(timeout_s=2.0)

        trim_roll = self.roll
        trim_pitch = self.pitch
        trim_yaw = self.yaw
        initial_speed = self.speed
        speed_delta = command_speed - initial_speed

        if abs(speed_delta) < 0.05:
            raise RuntimeError(
                "Commanded speed is too close to the current speed to measure a step."
            )

        threshold_speed = initial_speed + (response_fraction * speed_delta)

        print(
            f"Initial follower speed: {initial_speed:.2f} m/s. "
            f"Commanding {command_speed:.2f} m/s."
        )
        print(
            f"{response_fraction * 100:.0f}% threshold: "
            f"{threshold_speed:.2f} m/s."
        )

        csv_file = None
        writer = None
        if csv_path:
            csv_file = open(csv_path, "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "time_s",
                    "command_speed_m_s",
                    "actual_speed_m_s",
                    "threshold_speed_m_s",
                    "throttle_cmd",
                ]
            )

        start_time = time.monotonic()
        reached_time = None
        last_print_time = -1.0

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start_time
                if elapsed > duration_s:
                    break

                self._drain_messages()
                throttle_cmd = self.speed_to_throttle(command_speed)
                self.send_attitude_target(
                    trim_roll,
                    trim_pitch,
                    trim_yaw,
                    throttle_cmd,
                )

                if writer is not None:
                    writer.writerow(
                        [
                            elapsed,
                            command_speed,
                            self.speed,
                            threshold_speed,
                            throttle_cmd,
                        ]
                    )

                if elapsed - last_print_time >= 0.25:
                    print(
                        f"t={elapsed:6.3f}s | "
                        f"cmd={command_speed:6.2f} m/s | "
                        f"speed={self.speed:6.2f} m/s | "
                        f"thr={throttle_cmd:5.3f}"
                    )
                    last_print_time = elapsed

                if reached_time is None and self.reached_threshold(
                    self.speed,
                    initial_speed,
                    command_speed,
                    threshold_speed,
                ):
                    reached_time = elapsed
                    print(
                        f"\nReached {threshold_speed:.2f} m/s at "
                        f"{reached_time:.3f} seconds."
                    )

                time.sleep(sample_period)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if csv_file is not None:
                csv_file.close()

        if reached_time is None:
            print(
                f"\nDid not reach {threshold_speed:.2f} m/s within "
                f"{duration_s:.1f} seconds."
            )
        else:
            print(
                f"\nSpeed step response time to "
                f"{response_fraction * 100:.0f}% "
                f"({threshold_speed:.2f} m/s): {reached_time:.3f} seconds."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Command a follower speed step and measure time to 63% response "
            "from the current speed."
        )
    )
    parser.add_argument(
        "--follower-port",
        default="udp:127.0.0.1:14562",
        help="MAVLink connection string for the follower aircraft.",
    )
    parser.add_argument(
        "--command-speed",
        type=float,
        default=20.0,
        help="Follower target speed in m/s.",
    )
    parser.add_argument(
        "--response-fraction",
        type=float,
        default=0.63,
        help="Fraction of the speed step used for the response threshold.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=15.0,
        help="Maximum measurement duration in seconds.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Attitude command and measurement loop rate.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="Delay after switching to GUIDED mode before applying the step.",
    )
    parser.add_argument(
        "--csv",
        default="speed_step_response.csv",
        help="Optional CSV output path. Use an empty string to disable logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test = FollowerSpeedStepResponse(args.follower_port)
    test.run(
        command_speed=args.command_speed,
        response_fraction=args.response_fraction,
        duration_s=args.duration_s,
        rate_hz=args.rate_hz,
        settle_s=args.settle_s,
        csv_path=args.csv,
    )
