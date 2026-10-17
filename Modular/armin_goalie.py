"""
File purpose: TLDR - Goalie strategy: WorldState in, Intent out.

Description:
Strategy only; it never touches the motors. Each tick Goalie.step() reads a field-frame
WorldState (field_model.py) and returns an armin_motion.Intent, picking one behaviour:

  guard      stand on the penalty line where the ball->goal-centre line crosses it,
             facing the ball (slides sideways as the ball moves)
  intercept  the ball is rolling at our goal: go to where it will cross our line
  engage     a slow ball close in front of us: get on its goal side and push it
             upfield with the dribbler on (this is the "save and take possession" part)
  hold       ball is on the dribbler: stay put facing upfield; the RoleManager
             (armin_roles.py) sees has_ball and swaps us to striker. If no swap comes
             within hold_timeout (no radio), kick it upfield instead (clear)
  wait       teammate is partly in our penalty area: stay outside it (multiple defence)

Rules this respects (RCJ Soccer 2026):
  2.6.1  never fully inside the penalty area -> every target goes through legal_point()
  2.6.2  only one of our robots even partly in the penalty area -> 'wait'
  2.5.5 / 2.9.2  a robot that doesn't respond to the ball can be deemed damaged -> 'engage'

Field frame: cm, origin at the centre spot, +y toward the enemy goal (ours is at -y),
heading 0 = facing +y, CCW positive. The WorldState is duck-typed.
"""

import math
from typing import Optional, Tuple

from armin_config import GoalieConfig
from armin_motion import Intent, STOP

BALL_R = 2.1            # cm, 42 mm golf ball (2026 Vision and IR leagues)


class BallTracker:
    """Ball position memory and velocity from successive field-frame sightings."""

    def __init__(self, memory: float = 1.0, tau: float = 0.1, gap: float = 0.3) -> None:
        self.memory = memory        # s a sighting stays usable after the ball disappears
        self.tau = tau              # s velocity smoothing
        self.gap = gap              # s without a sighting before velocity restarts from zero
        self.pos = None
        self.vel = (0.0, 0.0)
        self.t = None

    def update(self, now: float, ball: Optional[Tuple[float, float]]) -> None:
        if ball is None:
            return
        if self.pos is not None and 0 < now - self.t < self.gap:
            dt = now - self.t
            a = dt / (self.tau + dt)
            vx, vy = (ball[0] - self.pos[0]) / dt, (ball[1] - self.pos[1]) / dt
            self.vel = (self.vel[0] + a * (vx - self.vel[0]), self.vel[1] + a * (vy - self.vel[1]))
        else:
            self.vel = (0.0, 0.0)
        self.pos, self.t = (float(ball[0]), float(ball[1])), now

    def get(self, now: float) -> Optional[Tuple[float, float]]:
        """Last known position, or None once it is older than ``memory``."""
        if self.pos is None or now - self.t > self.memory:
            return None
        return self.pos

    @property
    def speed(self) -> float:
        return math.hypot(*self.vel)


def heading_of(dx: float, dy: float) -> float:
    """Field heading (0 = +y, CCW+) of a direction."""
    return math.atan2(-dx, dy)


class Goalie:
    def __init__(self, config: GoalieConfig = None) -> None:
        self.config = config or GoalieConfig()
        self.ball = BallTracker()
        self.mode = 'guard'
        self.has_ball = False
        self._held_since = None

    # ---- geometry -------------------------------------------------------

    @property
    def guard_y(self) -> float:
        return self.config.penalty_front_y

    def legal_point(self, x: float, y: float) -> Tuple[float, float]:
        """Nearest point where our centre is at most max_intrusion past the penalty lines
        (rule 2.6.1: never fully inside)."""
        cfg = self.config
        edge_y = cfg.penalty_front_y - cfg.max_intrusion
        edge_x = cfg.penalty_half_width - cfg.max_intrusion
        if y >= edge_y or abs(x) >= edge_x:
            return x, y
        if edge_y - y <= edge_x - abs(x):
            return x, edge_y
        return math.copysign(edge_x, x if x else 1.0), y

    def in_penalty_area(self, x: float, y: float) -> bool:
        """True if a robot centred at (x, y) is at least partly in our penalty area."""
        cfg = self.config
        r = cfg.robot_radius
        return abs(x) < cfg.penalty_half_width + r and y < cfg.penalty_front_y + r

    def guard_point(self, ball) -> Tuple[float, float]:
        """Where the ball -> goal-centre line crosses our guard line."""
        cfg = self.config
        gy = self.guard_y
        if ball is None:
            return 0.0, gy
        bx, by = ball
        if by > gy + 1.0:
            x = bx * (gy - cfg.goal_line_y) / (by - cfg.goal_line_y)
        else:
            x = bx                                      # ball level with / behind our line
        return max(-cfg.guard_half_width, min(cfg.guard_half_width, x)), gy

    def crossing(self, ball) -> Optional[float]:
        """x where the rolling ball will cross our guard line, if it is coming at our goal
        soon enough to matter; None otherwise."""
        cfg = self.config
        bx, by = ball
        vx, vy = self.ball.vel
        if vy > -cfg.intercept_min_speed or by <= self.guard_y:
            return None
        t = (self.guard_y - by) / vy
        if t > cfg.intercept_horizon:
            return None
        x = bx + vx * t
        # wide of the goal by more than we could ever cover: let it go
        if abs(x) > cfg.goal_half_width + cfg.guard_half_width:
            return None
        return x

    def ball_on_dribbler(self, state, ball) -> bool:
        """Geometric possession check: ball just in front of our bumper, near centre-line."""
        cfg = self.config
        me = state.me
        dx, dy = ball[0] - me.x, ball[1] - me.y
        c, s = math.cos(me.heading), math.sin(me.heading)
        right, fwd = c * dx + s * dy, -s * dx + c * dy
        gap = fwd - cfg.robot_radius - BALL_R
        return -1.0 <= gap <= cfg.capture_distance and abs(right) <= cfg.capture_half_width

    def should_engage(self, ball) -> bool:
        """Ball slow and close in front of us. Once engaged, the zone grows by
        engage_release_margin so a ball nudged just past its edge isn't abandoned."""
        cfg = self.config
        bx, by = ball
        grow = cfg.engage_release_margin if self.mode == 'engage' else 0.0
        return (by < cfg.penalty_front_y + cfg.engage_depth + grow
                and abs(bx) < cfg.engage_half_width + grow
                and self.ball.speed < cfg.engage_max_speed)

    # ---- behaviours -----------------------------------------------------

    def engage(self, state, ball) -> Intent:
        """Get on the ball's goal side, then push it upfield with the dribbler running."""
        cfg = self.config
        me = state.me
        bx, by = ball
        ux, uy = bx, by - cfg.goal_line_y                # own goal centre -> ball
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        rx, ry = me.x - bx, me.y - by
        along = rx * ux + ry * uy                        # + = we are upfield of the ball
        side = -rx * uy + ry * ux                        # lateral offset from the line
        clearance = cfg.robot_radius + BALL_R + 3.0
        face = heading_of(ux, uy)
        behind = (bx - ux * cfg.approach_offset, by - uy * cfg.approach_offset)

        if math.dist(self.legal_point(*behind), behind) > 2.0:
            # ball deep in our penalty area: its goal side is out of bounds for us. Come in
            # facing it from where we are and let the dribbler's backspin pull it out.
            dx, dy = me.x - bx, me.y - by
            d = math.hypot(dx, dy) or 1.0
            touch = cfg.robot_radius + BALL_R - 1.0
            target = (bx + dx / d * touch, by + dy / d * touch)
            return Intent(target=self.legal_point(*target), face_point=ball, speed=0.3,
                          dribbler=True, label='engage:reach')
        facing = abs(math.atan2(math.sin(face - me.heading), math.cos(face - me.heading)))
        if (along < -cfg.robot_radius and abs(side) < cfg.capture_half_width + 3.0
                and facing < math.radians(cfg.push_heading_tolerance_deg)):
            # lined up behind the ball: drive through it
            target = (bx + ux * 10.0, by + uy * 10.0)
            return Intent(target=self.legal_point(*target), face=face, speed=0.6,
                          dribbler=True, label='engage:push')
        if along > -cfg.robot_radius and abs(side) < clearance:
            # on the wrong side, the straight path goes through the ball: step round it
            sgn = 1.0 if side >= 0 else -1.0
            # waypoint diagonally behind the ball, so the path on to 'behind' stays clear of it
            back, out = cfg.robot_radius + 4.0, clearance + 4.0
            target = (bx - ux * back - uy * sgn * out, by - uy * back + ux * sgn * out)
            return Intent(target=self.legal_point(*target), face_point=ball,
                          dribbler=True, label='engage:around')
        return Intent(target=self.legal_point(*behind), face=face, dribbler=True,
                      label='engage:line-up')

    def step(self, state, now: float, has_ball: Optional[bool] = None) -> Intent:
        """One tick. ``has_ball``: pass the camera's possession latch when you have it;
        otherwise possession is judged from the field geometry."""
        if state is None or not state.valid:
            self.mode = 'lost'
            return STOP
        self.ball.update(now, state.ball)
        ball = self.ball.get(now)
        cfg = self.config

        if has_ball is None:
            has_ball = ball is not None and self.ball_on_dribbler(state, ball)
        self.has_ball = bool(has_ball)

        mate = getattr(state, 'mate', None)
        if mate is not None and self.in_penalty_area(mate.x, mate.y):
            self.mode = 'wait'
            gx, _ = self.guard_point(ball)
            return Intent(target=(gx, cfg.penalty_front_y + cfg.robot_radius + cfg.mate_wait_gap),
                          face_point=ball, label='wait')

        if self.has_ball:
            if self._held_since is None:
                self._held_since = now
            if now - self._held_since < cfg.hold_timeout:
                self.mode = 'hold'
                return Intent(face=0.0, dribbler=True, label='hold')
            # nobody took over: clear it upfield once we face that way
            self.mode = 'clear'
            aimed = abs(state.me.heading) < math.radians(cfg.clear_heading_tolerance_deg)
            return Intent(face=0.0, dribbler=not aimed, kick=aimed, label='clear')
        self._held_since = None
        if ball is None:
            self.mode = 'guard'
            return Intent(target=self.guard_point(None), face=0.0, label='guard:no-ball')

        x = self.crossing(ball)
        if x is not None:
            self.mode = 'intercept'
            x = max(-cfg.guard_half_width, min(cfg.guard_half_width, x))
            return Intent(target=(x, self.guard_y), face_point=ball, speed=1.0, label='intercept')
        if self.should_engage(ball):
            self.mode = 'engage'
            return self.engage(state, ball)
        self.mode = 'guard'
        return Intent(target=self.legal_point(*self.guard_point(ball)), face_point=ball,
                      label='guard')
