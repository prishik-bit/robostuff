"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: Tracking the ball state

Description:
Keeps track of the ball's position relative to the robot, the robot's control state, and the goal's position.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple


@dataclass
class BallState:
    """Most recent ball observation in pixels relative to image centre."""

    offset: Optional[Tuple[int, int]] = None
    last_seen: float = 0.0


@dataclass
class ControlState:
    """Current driving mode, browser keys, and debug requests."""

    mode: str = 'manual'
    active_keys: Set[str] = field(default_factory=set)
    keys_last_seen: float = 0.0
    print_vector_requested: bool = False
    debug_motor: bool = False
    take_photo : bool = False

@dataclass
class GoalState:
    offset: Optional[Tuple[int, int]] = None
    last_seen: float = 0.0

@dataclass
class RobotState:
    """All mutable state shared by camera, motor, and WebSocket tasks."""

    ball: BallState = field(default_factory=BallState)
    control: ControlState = field(default_factory=ControlState)
    goal: GoalState = field(default_factory=GoalState)
    clients: Set[object] = field(default_factory=set)
    is_running: bool = True
    has_possession : bool = False
    has_orbited : bool = False
    is_goalie : bool = False
