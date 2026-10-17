"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: TLDR - A Settings Menu

Description:
This file contains many data classes that store variables used througout this project.
This is a centralized place to have these variables to make it easier to change them in the future.
The variables are grouped into different classes based on their purpose.

"""

from dataclasses import dataclass, field
from tof import TOFChain
from typing import Tuple


@dataclass(frozen=True)
class MotorConfig:
    """Motor addresses, calibration, speed limits, and dribbler settings.

    The four drive motors are ordered ``FR, BR, BL, FL`` to match Hari's
    wiring and the index assumptions in ``MotorController.move``.
    """

    addresses: Tuple[int, ...] = (25, 26, 27, 28)
    # Each calibration is (electrical angle offset, sin/cos centre).
    calibrations: Tuple[Tuple[int, int], ...] = (
        (1133688832, 1242),
        (944037888, 1246),
        (1327844608, 1233),
        (1577042432, 1253),
    )
    max_speed: int = 100_000_000
    
    enable_dribbler: bool = True
    dribbler_address: int = 29
    dribbler_calibration: Tuple[int, int] = (1439443968, 1248)
    dribbler_speed: int = -250_000_000
    
    orbit_standoff_radius: int = 170
    orbit_arrived_angle_tolerance: float = 1
    orbit_arrived_radius_tolerance_ratio: float = 0.10
    orbit_full_speed_angle: float = 25.0
    orbit_max_tangential_speed_ratio: float = 0.5
    orbit_max_radial_speed_ratio: float = 0.3
    orbit_radial_gain: float = 300_000.0
    
    drive_to_ball_speed_ratio: float = 1.0
    goal_search_speed_ratio: float = 0.2
    goal_rotation_speed_ratio: float = 0.3
    goal_align_tolerance_degrees: float = 2

    # The previous version wrapped each pair in MotorCalibration objects:
    # calibrations = (
    #     MotorCalibration(1327731200, 1241),
    #     MotorCalibration(1435147520, 1243),
    # )
    # The tuple-of-tuples above is intentionally simpler for now.

@dataclass
class VisionConfig:
    """HSV, geometry, camera-angle, timeout, and CLAHE tuning values."""

    # HSV bounds are kept together so calibration is easy to read and edit.
    # ball_lower: Tuple[int, int, int] = (0, 200, 77)
    # ball_upper: Tuple[int, int, int] = (15, 255, 255)
    
    # Values for in-house testing
    ball_lower: Tuple[int, int, int] = (0, 180, 70)
    ball_upper: Tuple[int, int, int] = (23, 255, 255)
    
    min_contour_area: int = 1
    dead_zone_radius: int = 132
    ball_dribble_radius: int = 145
    orbit_radius: int = 190
    camera_rotation_offset: float = 0
    ball_lost_timeout: float = 0.5
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    debug_mask : bool = False
    valid_mask_path: str = 'bot_mask.png'
    
    last_ball_vector : Tuple[float, float] = None
    last_ball_distance : float = None
    
    # yellow_goal_lower: Tuple[int, int, int] = (20, 235, 100)
    # yellow_goal_upper: Tuple[int, int, int] = (40, 255, 255)
    
    # blue_goal_lower: Tuple[int, int, int] = (95, 207, 60)
    # blue_goal_upper: Tuple[int, int, int] = (105, 255, 100)
    
    yellow_goal_lower: Tuple[int, int, int] = (20, 180, 30)
    yellow_goal_upper: Tuple[int, int, int] = (40, 255, 255)
    
    blue_goal_lower: Tuple[int, int, int] = (95, 200, 160)
    blue_goal_upper: Tuple[int, int, int] = (120, 255, 255)

    goal_stop_distance : int = 190
    goal_min_contour_area : int = 120  # goals are big; a larger floor rejects speckle
    
    last_goal_vector : Tuple[float, float] = None
    last_goal_distance : float = None
    
    in_range_for_kick: float = 118.7
    
    goals : Tuple[str, str] = ('yellow_goal', 'blue_goal')
    
    target_goal : str = goals[1]  # 'yellow_goal' or 'blue_goal' - the goal we attack
    opposite_goal : str = goals[1] if target_goal == goals[0] else goals[0]
    

@property
def opposite_goal(self) -> str:
    return self.goals[1] if self.target_goal == self.goals[0] else self.goals[0]

@dataclass(frozen=True)
class CameraConfig:
    """Capture format and manual image controls requested from Picamera2."""

    size: Tuple[int, int] = (640, 480)
    fps: int = 60
    exposure_time: int = 15_000 # 66656
    analogue_gain: float = 16
    colour_gains: Tuple[float, float] = (2.4364535808563232, 1.9698092937469482)

    @property
    def frame_duration_us(self) -> int:
        return int(1_000_000 / self.fps)

@dataclass
class TOFConfig:
    ADDRESS_SET_1 = [0x51, 0x52, 0x53, 0x50, 0x55, 0x56, 0x57, 0x5a] 
    TOF_CHAIN_1 = TOFChain(ADDRESS_SET_1)
    


@dataclass(frozen=True)
class NetworkConfig:
    """Network endpoint used by the browser control page."""

    websocket_port: int = 8765


@dataclass(frozen=True)
class ControlConfig:
    """Timing and physical control-input settings."""

    key_lost_timeout: float = 0.3
    motor_loop_delay: float = 0.02
    # camera_loop_delay: float = 0.01
    debug_motor: bool = True
    debug_print_hz: int = 5
    movement_switch_gpio: int = 26
    goal_switch_gpio : int = 25
    manual_translation_speed_ratio: float = 0.7
    manual_rotation_speed_ratio: float = 0.3
    manual_keys: dict[str, int] = field(
        default_factory=lambda: {'w': 0, 'd': 90, 's': 180, 'a': 270}
    )
    rotation_keys: dict[str, int] = field(
        default_factory=lambda: {'q': 1, 'e': -1}
    )
    miscellaneous_keys: dict[str, str] = field(
        default_factory=lambda: {'k': 'dribble', 'x': 'orbit'}
    )

    @property
    def dribble_key(self) -> str:
        return next(key for key, action in self.miscellaneous_keys.items() if action == 'dribble')

    @property
    def orbit_key(self) -> str:
        return next(key for key, action in self.miscellaneous_keys.items() if action == 'orbit')


@dataclass(frozen=True)
class ImuConfig:
    """I2C address and axis convention for the optional IMU."""

    address: int = 0x4A

@dataclass(frozen=True)
class LidarConfig:
    """UART port, filtering, and mounting for the LDRobot D500 (LD19) lidar."""

    port: str = '/dev/serial0'  # Pi GPIO14/15 UART; a USB adapter shows up as /dev/ttyUSB0
    baudrate: int = 230_400
    min_distance_mm: int = 20      # D500 rated range is 0.02 m - 12 m
    max_distance_mm: int = 12_000
    min_intensity: int = 100       # weaker returns are usually noise or glancing hits
    stale_after: float = 0.3       # seconds before a bin's reading is treated as unknown
    mounting_offset: float = 0.0   # degrees to add so lidar 0 lines up with robot forward

@dataclass(frozen=True)
class MotionConfig:
    """Field-frame motion control used by armin_motion.MotionController (cm, degrees)."""

    field_half_x: float = 82.0          # targets are clamped this far from the centre spot
    field_half_y: float = 112.0         # (walls are at 91 / 121.5; leaves a robot radius)
    arrive_radius: float = 3.0          # cm: closer than this counts as arrived
    slow_radius: float = 30.0           # cm: start ramping speed down inside this
    min_speed_ratio: float = 0.15       # ramp floor so the bot doesn't stall short of target
    heading_tolerance_deg: float = 3.0
    heading_full_speed_deg: float = 60.0  # full rotation speed beyond this heading error
    max_rotation_ratio: float = 0.6     # of max_speed
    kick_cooldown: float = 0.5          # s between solenoid kicks

@dataclass(frozen=True)
class GoalieConfig:
    """Goalie geometry and thresholds for armin_goalie (field cm, our goal at -y).

    Penalty area size is not in the 2026 rules text; measure it on the real field.
    Rule 2.6.1: a robot may never be FULLY inside the penalty area, so the goalie's
    centre is kept within max_intrusion of the penalty line (most of the body outside).
    """

    robot_radius: float = 9.0
    goal_line_y: float = -109.5         # our goal mouth (front face of the posts)
    goal_half_width: float = 30.0       # posts at |x| = 30
    penalty_front_y: float = -84.5      # front line of our penalty area (goal line + 25)
    penalty_half_width: float = 40.0    # MEASURE: penalty area is ~80 cm wide
    max_intrusion: float = 4.0          # cm the centre may go past the penalty line
    guard_half_width: float = 38.0      # how far sideways the goalie slides on its line
    engage_depth: float = 35.0          # go for a ball within this far in front of the penalty line
    engage_half_width: float = 55.0     # ...and within this |x|
    engage_max_speed: float = 60.0      # cm/s: faster balls are blocked, not chased
    engage_release_margin: float = 20.0 # cm the engage zone grows by once we've committed
    intercept_min_speed: float = 25.0   # cm/s toward our goal before we predict a crossing
    intercept_horizon: float = 1.5      # s ahead we bother predicting
    approach_offset: float = 14.0       # cm behind the ball (goal side) to line up before pushing
    push_heading_tolerance_deg: float = 25.0  # only push once the dribbler faces this close to the push line
    capture_distance: float = 4.0       # cm gap between bumper and ball that counts as "on the dribbler"
    capture_half_width: float = 5.0     # cm either side of centre-line for a geometric capture
    mate_wait_gap: float = 12.0         # cm outside the penalty line to wait while the mate is in it
    hold_timeout: float = 1.0           # s holding the ball with no role swap before we clear it
    clear_heading_tolerance_deg: float = 20.0  # kick to clear only when facing this close to upfield

@dataclass(frozen=True)
class RoleConfig:
    """Goalie/striker swapping between the two robots (see armin_roles)."""

    robot_id: int = 1                   # 1 or 2; the other robot gets the other id
    default_role: str = 'striker'       # role at power-on and whenever the radio is lost
    possession_confirm: float = 0.25    # s the goalie must hold the ball before claiming striker
    swap_cooldown: float = 1.0          # s after a role change before another voluntary change
    conflict_timeout: float = 0.3       # s both robots may claim the same role before tie-break
    mate_stale: float = 0.5             # s without a heartbeat = no radio (match field_model.MATE_STALE)

@dataclass(frozen=True)
class SolenoidConfig:
    """GPIO address and active time for solenoid"""
    GPIOpin: int = 17
    active_time: float = 0.8
    
@dataclass(frozen=True)
class RobotConfig:
    """Complete startup configuration passed into ``HariApplication``."""

    motors: MotorConfig = field(default_factory=MotorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    goalie: GoalieConfig = field(default_factory=GoalieConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)
    solenoid: SolenoidConfig = field(default_factory=SolenoidConfig)