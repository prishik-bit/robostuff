"""
File purpose: TLDR - Turns a field-frame Intent from strategy into motor commands.

Description:
Strategy code reads a WorldState (field_model.py) and returns an Intent: where to be,
which way to face, dribbler on/off, kick or not. It never calls the motors itself.
MotionController.execute() is the one place that converts that field-frame request into
the robot-frame calls MotorController understands.

Conventions (match field_model / lidar_loc):
  field     cm, origin at the centre spot, +x right, +y toward the ENEMY goal
  heading   rad, 0 = facing the enemy goal (+y), counter-clockwise positive

MotorController speaks a different dialect, converted here and nowhere else:
  rotate_and_move bearing   degrees, 0 = ahead, CLOCKWISE positive
  eased_rotation            raw speed units, positive turns CCW

The WorldState is duck-typed (needs .valid and .me.x/.y/.heading) so this module does not
import field_model and can be exercised off the Pi with a fake motor object.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from armin_config import MotionConfig


def wrap(a: float) -> float:
    """Angle to -pi..pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Intent:
    """What strategy wants this tick. Everything is in the field frame.

    target      (x, y) cm to drive to; None = no translation
    face        heading (rad, 0 = enemy goal, CCW+) to turn to; None = no rotation
    face_point  (x, y) cm to turn towards (e.g. the ball); used when face is None
    speed       0..1 cap on translation speed, as a ratio of max_speed
    dribbler    dribbler on/off
    kick        fire the solenoid (rate-limited by MotionConfig.kick_cooldown)
    label       short name of the behaviour, for debug prints
    """

    target: Optional[Tuple[float, float]] = None
    face: Optional[float] = None
    face_point: Optional[Tuple[float, float]] = None
    speed: float = 1.0
    dribbler: bool = False
    kick: bool = False
    label: str = ''


STOP = Intent(label='stop')


class MotionController:
    """Executes Intents against a MotorController (and optionally a Solenoid)."""

    def __init__(self, motors, config: MotionConfig = None, solenoid=None) -> None:
        self.motors = motors
        self.config = config or MotionConfig()
        self.solenoid = solenoid
        self._last_kick = -math.inf

    # ---- frame conversions ----------------------------------------------

    @staticmethod
    def to_robot(me, x: float, y: float) -> Tuple[float, float]:
        """Field point -> robot frame (right, forward) cm."""
        dx, dy = x - me.x, y - me.y
        c, s = math.cos(me.heading), math.sin(me.heading)
        return c * dx + s * dy, -s * dx + c * dy

    @staticmethod
    def motor_bearing(right: float, forward: float) -> float:
        """Robot-frame vector -> rotate_and_move bearing (deg, 0 = ahead, clockwise+)."""
        return math.degrees(math.atan2(right, forward)) % 360

    def clamp_target(self, target: Tuple[float, float]) -> Tuple[float, float]:
        """Keep a target inside the field so we never drive into a wall."""
        cfg = self.config
        return (min(max(target[0], -cfg.field_half_x), cfg.field_half_x),
                min(max(target[1], -cfg.field_half_y), cfg.field_half_y))

    # ---- the two control terms ------------------------------------------

    def translation(self, me, intent: Intent) -> Tuple[float, int]:
        """(bearing deg, speed) toward intent.target; speed 0 once arrived.
        Full speed outside slow_radius, ramping down to min_speed_ratio at arrive_radius."""
        if intent.target is None:
            return 0.0, 0
        cfg = self.config
        x, y = self.clamp_target(intent.target)
        right, forward = self.to_robot(me, x, y)
        dist = math.hypot(right, forward)
        if dist <= cfg.arrive_radius:
            return 0.0, 0
        ramp = (dist - cfg.arrive_radius) / max(cfg.slow_radius - cfg.arrive_radius, 1e-6)
        ratio = max(min(ramp, 1.0), cfg.min_speed_ratio) * min(max(intent.speed, 0.0), 1.0)
        return self.motor_bearing(right, forward), int(self.motors.config.max_speed * ratio)

    def desired_heading(self, me, intent: Intent) -> Optional[float]:
        if intent.face is not None:
            return intent.face
        if intent.face_point is not None:
            dx, dy = intent.face_point[0] - me.x, intent.face_point[1] - me.y
            if math.hypot(dx, dy) > self.config.arrive_radius:
                return math.atan2(-dx, dy)
        return None

    def rotation(self, me, intent: Intent) -> int:
        """eased_rotation (positive = CCW) that turns us toward the desired heading."""
        want = self.desired_heading(me, intent)
        if want is None:
            return 0
        cfg = self.config
        err = math.degrees(wrap(want - me.heading))    # + means we must turn CCW
        if abs(err) <= cfg.heading_tolerance_deg:
            return 0
        ease = min(abs(err) / cfg.heading_full_speed_deg, 1.0)
        return int(math.copysign(self.motors.config.max_speed * cfg.max_rotation_ratio * ease, err))

    def heading_error(self, state, intent: Intent) -> Optional[float]:
        """Degrees still to turn (+ = CCW), or None when the Intent has no heading."""
        want = self.desired_heading(state.me, intent)
        return None if want is None else math.degrees(wrap(want - state.me.heading))

    def arrived(self, state, intent: Intent) -> bool:
        """True once inside arrive_radius of the (clamped) target, or there is no target."""
        if intent.target is None:
            return True
        x, y = self.clamp_target(intent.target)
        return math.hypot(x - state.me.x, y - state.me.y) <= self.config.arrive_radius

    # ---- execute --------------------------------------------------------

    def execute(self, state, intent: Intent, now: float = None) -> bool:
        """Apply one tick of ``intent``. Returns False (and stops the wheels) when the
        world state is not trustworthy, so the caller can fall back to blind behaviour."""
        motors = self.motors
        if state is None or not state.valid or state.me.heading is None:
            motors.stop_wheels()
            return False

        bearing, speed = self.translation(state.me, intent)
        rotation = self.rotation(state.me, intent)
        if speed == 0 and rotation == 0:
            motors.stop_wheels()
        else:
            motors.rotate_and_move(bearing, speed, rotation)
        motors.spin_dribbler(intent.dribbler)

        if intent.kick:
            self.kick(now)

        motors.debug(f"[motion] {intent.label or '-'} bearing={bearing:.0f} "
                     f"speed={speed} rot={rotation}")
        return True

    def kick(self, now: float = None) -> bool:
        """Fire the solenoid unless it fired within kick_cooldown. Returns True if fired."""
        now = time.monotonic() if now is None else now
        if self.solenoid is None or now - self._last_kick < self.config.kick_cooldown:
            return False
        self._last_kick = now
        self.solenoid.kick()
        return True
