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


class FollowerRollStepResponse:
    def __init__(self, follower_port):
        print(f"Connecting to Follower on {follower_port}...")
        self.follower = mavutil.mavlink_connection(follower_port)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_attitude_time = None

        self.follower.wait_heartbeat()
        print("Follower heartbeat verified.")
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

    def _drain_attitude_messages(self):
        updated = False
        while True:
            msg = self.follower.recv_msg()
            if msg is None:
                break
            if msg.get_type() == "ATTITUDE":
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
                self.last_attitude_time = time.monotonic()
                updated = True
        return updated

    def wait_for_attitude(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._drain_attitude_messages():
                return
            time.sleep(0.02)
        raise TimeoutError("Timed out waiting for follower ATTITUDE messages.")

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

    def run(
        self,
        command_roll_deg,
        threshold_deg,
        thrust,
        duration_s,
        rate_hz,
        settle_s,
        csv_path,
    ):
        command_roll = math.radians(command_roll_deg)
        threshold_roll = math.radians(threshold_deg)
        sample_period = 1.0 / rate_hz

        self.wait_for_attitude(timeout_s=5.0)
        trim_pitch = self.pitch
        trim_yaw = self.yaw
        initial_roll_deg = math.degrees(self.roll)

        print(
            f"Initial follower roll: {initial_roll_deg:.2f} deg. "
            f"Commanding {command_roll_deg:.2f} deg."
        )
        print(f"63% threshold: {threshold_deg:.2f} deg.")

        self.set_follower_mode("GUIDED")
        time.sleep(settle_s)

        csv_file = None
        writer = None
        if csv_path:
            csv_file = open(csv_path, "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(["time_s", "command_roll_deg", "actual_roll_deg"])

        start_time = time.monotonic()
        reached_time = None
        last_print_time = -1.0

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start_time
                if elapsed > duration_s:
                    break

                self._drain_attitude_messages()
                self.send_attitude_target(command_roll, trim_pitch, trim_yaw, thrust)

                actual_roll_deg = math.degrees(self.roll)
                if writer is not None:
                    writer.writerow([elapsed, command_roll_deg, actual_roll_deg])

                if elapsed - last_print_time >= 0.25:
                    print(
                        f"t={elapsed:6.3f}s | "
                        f"cmd={command_roll_deg:6.2f} deg | "
                        f"roll={actual_roll_deg:6.2f} deg"
                    )
                    last_print_time = elapsed

                if reached_time is None and self.roll >= threshold_roll:
                    reached_time = elapsed
                    print(
                        f"\nReached {threshold_deg:.2f} deg at "
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
                f"\nDid not reach {threshold_deg:.2f} deg within "
                f"{duration_s:.1f} seconds."
            )
        else:
            print(
                f"\nRoll step response time to 63% "
                f"({threshold_deg:.2f} deg): {reached_time:.3f} seconds."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command a follower roll step and measure time to 63% response."
    )
    parser.add_argument(
        "--follower-port",
        default="udp:127.0.0.1:14562",
        help="MAVLink connection string for the follower aircraft.",
    )
    parser.add_argument(
        "--command-roll-deg",
        type=float,
        default=20.0,
        help="Follower roll step command in degrees.",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=12.6,
        help="Measured roll threshold in degrees.",
    )
    parser.add_argument(
        "--thrust",
        type=float,
        default=0.5,
        help="Normalized thrust command sent with the attitude target.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
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
        default="roll_step_response.csv",
        help="Optional CSV output path. Use an empty string to disable logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test = FollowerRollStepResponse(args.follower_port)
    test.run(
        command_roll_deg=args.command_roll_deg,
        threshold_deg=args.threshold_deg,
        thrust=max(0.0, min(1.0, args.thrust)),
        duration_s=args.duration_s,
        rate_hz=args.rate_hz,
        settle_s=args.settle_s,
        csv_path=args.csv,
    )
