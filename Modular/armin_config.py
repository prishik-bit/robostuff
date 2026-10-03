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
    dribbler_calibration: Tuple[int, int] = (1433137664, 1249)
    dribbler_speed: int = -250_000_000
    
    orbit_standoff_radius: int = 165
    orbit_arrived_angle_tolerance: float = 1.0
    orbit_arrived_radius_tolerance_ratio: float = 0.15
    orbit_full_speed_angle: float = 15.0
    orbit_max_tangential_speed_ratio: float = 0.5
    orbit_max_radial_speed_ratio: float = 0.3
    orbit_radial_gain: float = 400_000.0
    
    drive_to_ball_speed_ratio: float = 1.0
    goal_search_speed_ratio: float = 0.2
    goal_rotation_speed_ratio: float = 0.3
    goal_align_tolerance_degrees: float = 5

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
    ball_lower: Tuple[int, int, int] = (0, 200, 70)
    ball_upper: Tuple[int, int, int] = (15, 255, 255)
    
    min_contour_area: int = 1
    dead_zone_radius: int = 132
    ball_dribble_radius: int = 147
    orbit_radius: int = 190
    camera_rotation_offset: float = 0 #-90.0
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
    
    blue_goal_lower: Tuple[int, int, int] = (95, 231, 160)
    blue_goal_upper: Tuple[int, int, int] = (120, 255, 200)

    goal_stop_distance : int = 200
    goal_min_contour_area : int = 120  # goals are big; a larger floor rejects speckle
    
    last_goal_vector : Tuple[float, float] = None
    last_goal_distance : float = None
    
    in_range_for_kick: float = 118.7
    
    goals : Tuple[str, str] = ('yellow_goal', 'blue_goal')
    
    target_goal : str = goals[0]  # 'yellow_goal' or 'blue_goal' - the goal we attack
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
    pass


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
    # movement_switch_gpio: int = 26
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
    solenoid: SolenoidConfig = field(default_factory=SolenoidConfig)