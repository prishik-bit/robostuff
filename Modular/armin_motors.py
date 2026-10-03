"""Motor setup and movement decisions for Armin's chassis.

This is the only module that knows the BLDC driver API and the four-wheel
equations. Higher-level code asks it to move, spin, or stop.
"""

import math
import time
from typing import Iterable

import board
import busio

from armin_config import ControlConfig, MotorConfig, VisionConfig
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

class MotorController:
    """Own the initialised drive motors and turn decisions into wheel speeds.

    Wheel order is FR, BR, BL, FL; do not reorder without redoing the maths.
    """

    def __init__(
        self,
        config: MotorConfig,
        vision_config: VisionConfig,
        control_state,
        robot_state,
        control_config: ControlConfig = None,
        debug_hz: int = 5,
    ):
        self.config = config
        self.vision_config = vision_config
        self.control_config = control_config or ControlConfig()
        self.control_state = control_state
        self.robot_state = robot_state
        self.motors = []
        self.dribbler_motor = None
        self.debug_hz = debug_hz
        self._debug_last_print = 0.0

    # ---- setup ----------------------------------------------------------

    def setup(self) -> None:
        """Create and configure the drive drivers and the optional dribbler."""
        i2c = busio.I2C(board.SCL, board.SDA)
        for index, address in enumerate(self.config.addresses):
            self.motors.append(
                self._create_motor(i2c, address, self.config.calibrations[index])
            )
            print(f"Motor {index} ready")

        if self.config.enable_dribbler:
            self.dribbler_motor = self._create_motor(
                i2c, self.config.dribbler_address, self.config.dribbler_calibration
            )
            print("Dribbler motor ready")
        print("All motors ready")

    def _create_motor(self, i2c, address, calibration):
        """Create one driver in FOC speed mode, stopped."""
        elec_angle_offset, sincos_centre = calibration
        motor = PowerfulBLDCDriver(i2c, address)
        motor.set_current_limit_foc(166340)
        motor.set_id_pid_constants(1500, 200)
        motor.set_iq_pid_constants(1500, 200)  # was missing
        motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        motor.set_speed_limit(self.config.max_speed * 2)  # was missing; verify
        motor.set_ELECANGLEOFFSET(elec_angle_offset)
        motor.set_SINCOSCENTRE(sincos_centre)
        motor.configure_operating_mode_and_sensor(3, 1)  # FOC + sin/cos encoder
        motor.configure_command_mode(12)                 # speed command mode
        motor.set_speed(0)
        return motor

    # ---- low-level output -----------------------------------------------

    def _set_wheels(self, fr: int, br: int, bl: int, fl: int) -> None:
        for motor, speed in zip(self.motors, (fr, br, bl, fl)):
            motor.set_speed(int(speed))

    def spin_dribbler(self, engaged: bool) -> None:
        """Turn the dribbler on or off, if one exists."""
        
        if not self.config.enable_dribbler or self.dribbler_motor is None:
            return
        self.dribbler_motor.set_speed(self.config.dribbler_speed if engaged else 0)

    def stop_wheels(self) -> None:
        """Zero the drive wheels only; the dribbler keeps its state."""
        self._set_wheels(0, 0, 0, 0)

    def stop(self) -> None:
        """Stop every motor including the dribbler."""
        self.stop_wheels()
        self.spin_dribbler(False)

    # ---- movement primitives --------------------------------------------

    def rotate_and_move(self, degree: float, speed: float = None,
                        eased_rotation: float = 0) -> None:
        """Translate along a bearing (0 = ahead, clockwise positive) while
        adding a rotation term. Positive ``eased_rotation`` turns CCW."""
        speed = self.config.max_speed if speed is None else speed
        angle = math.radians(degree + 90)
        x = math.floor(math.cos(angle) * speed)
        y = math.floor(math.sin(angle) * speed)
        r = int(eased_rotation)
        self._set_wheels(y + x - r, y - x - r, -(y + x) - r, -(y - x) - r)

    def move(self, degree: float, speed: int = None) -> None:
        """Pure translation along a bearing."""
        self.rotate_and_move(degree, speed)

    def spin(self, speed: int) -> None:
        """Rotate in place. Positive is clockwise."""
        speed = int(speed)
        self._set_wheels(speed, speed, speed, speed)

    @staticmethod
    def clamp(val, min_val, max_val):
        return max(min_val, min(val, max_val))

    def spin_to_bearing(self, current_bearing: float, target_bearing: float,
                        tolerance: float = None) -> bool:
        """Rotate in place until ``current_bearing`` matches ``target_bearing``.
        Returns True once within tolerance (and stopped)."""
        tolerance = (self.config.orbit_arrived_angle_tolerance
                     if tolerance is None else tolerance)
        error = ((current_bearing - target_bearing + 180) % 360) - 180  # + means CW

        if abs(error) <= tolerance:
            self.stop_wheels()
            self._debug(f"[spin] arrived at {target_bearing:.1f} (err {error:.1f})")
            return True

        ease = min(abs(error) / self.config.orbit_full_speed_angle, 1.0)
        speed = math.copysign(self.config.max_speed * 0.3 * ease, error)
        self._debug(f"[spin] target={target_bearing:.1f} err={error:.1f} -> spin({int(speed)})")
        self.spin(speed)
        return False

    # ---- manual control -------------------------------------------------

    def apply_manual_keys(self, keys: Iterable[str]) -> None:
        """Rotate keys win, then orbit, then translation, then stop."""
        keys = set(keys)
        cfg = self.control_config
        dribble = cfg.dribble_key in keys

        for key, sign in cfg.rotation_keys.items():
            if key in keys:
                speed = sign * int(self.config.max_speed * cfg.manual_rotation_speed_ratio)
                self._debug(f"[manual] {sorted(keys)} -> spin({speed})")
                self.spin(speed)
                self.spin_dribbler(dribble)
                return

        if cfg.orbit_key in keys:
            self._debug(f"[manual] orbit -> arrived={self.orbit_to_behind_ball()}")
            return

        for key, degree in cfg.manual_keys.items():
            if key in keys:
                speed = int(self.config.max_speed * cfg.manual_translation_speed_ratio)
                self._debug(f"[manual] {sorted(keys)} -> move({degree}, {speed})")
                self.move(degree, speed)
                self.spin_dribbler(dribble)
                return

        self._debug(f"[manual] {sorted(keys)} -> no recognised key -> stop()")
        self.stop()

    def drive_to_the_ball(self, ball_visible: bool, ball_angle: float,
                          distance: float, eased_rotation: float = 0.0) -> None:
        """Stop if the ball is lost or already held; otherwise drive at it."""
        if not ball_visible:
            self.stop()
            return
        if self.robot_state.has_possession:
            self._debug("[auto] has possession -> stop_wheels()")
            self.stop_wheels()
            return

        speed = int(self.config.max_speed * self.config.drive_to_ball_speed_ratio)
        self._debug(f"[auto] ball {ball_angle:.1f}deg, {distance:.1f}px -> move at {speed}")
        self.rotate_and_move(ball_angle, speed, eased_rotation)

    def orbit_to_behind_ball(self, target_bearing: float = 0.0,
                             eased_rotation: float = 0.0) -> bool:
        """Sweep around the ball until it lies on ``target_bearing`` (robot frame,
        0 = ahead). Returns True once arrived."""
        offset = self.robot_state.ball.offset
        if offset is None:
            self.stop()
            return False

        cfg = self.config
        dx, dy = offset
        distance = math.hypot(dx, dy)
        ball_bearing = (
            math.degrees(math.atan2(dx, -dy)) + self.vision_config.camera_rotation_offset
        ) % 360
        angular_error = ((ball_bearing - target_bearing + 180) % 360) - 180  # signed
        standoff_radius = cfg.orbit_standoff_radius

        # if abs(angular_error) <= 10:
        #     standoff_radius = cfg.orbit_standoff_radius * 0.98
        
        if (abs(angular_error) <= cfg.orbit_arrived_angle_tolerance
                and abs(distance - standoff_radius) <= standoff_radius * cfg.orbit_arrived_radius_tolerance_ratio):
            self.stop_wheels()
            self._debug("Arrived behind ball")
            return True

        # Radial component: hold the standoff radius.
        max_radial = cfg.max_speed * cfg.orbit_max_radial_speed_ratio
        radial_speed = self.clamp((distance - standoff_radius) * cfg.orbit_radial_gain,
                                  -max_radial, max_radial)

        # Tangential component: sweep sideways, easing off near the target angle.
        tangent_bearing = ball_bearing + (90 if angular_error > 0 else -90)
        ease = min(abs(angular_error) / cfg.orbit_full_speed_angle, 1.0)
        tangent_speed = cfg.max_speed * cfg.orbit_max_tangential_speed_ratio * ease

        def get_unit_vector(bearing):  # bearing -> (x, y) unit vector, y down like the image
            return math.sin(math.radians(bearing)), -math.cos(math.radians(bearing))

        tdx, tdy = get_unit_vector(tangent_bearing)
        rdx, rdy = get_unit_vector(ball_bearing)
        vx = tangent_speed * tdx + radial_speed * rdx
        vy = tangent_speed * tdy + radial_speed * rdy

        final_bearing = math.degrees(math.atan2(vx, -vy)) % 360
        final_speed = int(min(math.hypot(vx, vy), cfg.max_speed))
        self.rotate_and_move(final_bearing, final_speed, eased_rotation)
        return False

    def drive_to_goal(self, goal_angle: float, eased_rotation: float = 0.0) -> None:
        """Drive toward the goal with the dribbler on."""
        self.rotate_and_move(goal_angle, self.config.max_speed, eased_rotation)
        self.spin_dribbler(True)

    # ---- debug ----------------------------------------------------------

    def debug(self, message: str) -> None:
        self._debug(message)

    def _debug(self, message: str) -> None:
        """Print at most ``debug_hz`` lines per second, when enabled."""
        if not self.control_state.debug_motor:
            return
        now = time.time()
        if now - self._debug_last_print < 1.0 / self.debug_hz:
            return
        self._debug_last_print = now
        print(message)