"""
field_model.py — the one thing strategy code reads: where everyone is on the field.

Every control tick, WorldModel.update() turns the Localiser (self pose), the EnemyTracker
(enemies) and the teammate's radio heartbeat into one WorldState snapshot, all in the same
field coordinates and on the same grid:

  field coordinates  cm, origin at the centre spot, +x right, +y toward the ENEMY goal
  heading            rad, 0 = facing the enemy goal (+y), counter-clockwise positive
                     (heading_deg gives degrees, -180..180)
  grid               the field inside the walls cut into COLS x ROWS cells (~10 cm);
                     col 0 is the left wall (-x), row 0 is OUR goal's end (-y)

    state = wm.update(now, ball=(bx, by))       # ball from the camera, field cm, or None
    state.me.x, state.me.y, state.me.heading, state.me.cell
    state.mate                                   # teammate (radio), or None
    state.enemies                                # 0-2 RobotState, best first
    state.enemies[0].heading                     # direction of travel, None when not moving
    dist, bearing = state.me_to(x, y)            # bearing: robot frame, 0 = ahead, CCW+
    grid = state.occupancy(inflate=5)            # rows x cols: 0 free, 1 me, 2 mate, 3 enemy
    state.tactics['shot_lane_clear']             # enemy_tracker.world_model_fields(...)

On the robot (each of the two runs this):
    loc = lidar_loc.Localiser(...)
    wm = WorldModel(loc, robot_id=1)
    imu / odometry loop  : loc.on_odometry(t, vx_body, vy_body, yaw)
    lidar UART reader    : loc.on_lidar_bytes(data, t)
    radio receive        : wm.on_heartbeat(payload, t)
    every 20-50 ms       : state = wm.update(t, ball); radio.send(wm.heartbeat()); strategy(state)
Use one monotonic clock (time.monotonic()) for every t on a robot.

Enemy heading: the lidar sees a robot's footprint, not which way it faces, so heading is the
direction of travel while it moves faster than HEADING_MIN_SPEED (slower, the direction is
mostly noise), held for HEADING_HOLD_S after it slows, then None. Expect ~20 deg typical error:
the tracker sees each enemy ~10x a second. A robot pushing sideways or reversing has a
direction of travel that isn't where it faces.
"""
import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import enemy_tracker as ET
import lidar_loc as L

# =============================== CONFIG ==================================
CELL = 10.0                     # cm target grid cell size (exact size below fits the walls)
COLS = int(round(2 * L.WALL_X / CELL))
ROWS = int(round(2 * L.WALL_Y / CELL))
CELL_W = 2 * L.WALL_X / COLS    # cm, actual cell width  (x)
CELL_H = 2 * L.WALL_Y / ROWS    # cm, actual cell height (y)
ROBOT_R = ET.ROBOT_R            # cm robot footprint radius used for occupancy
SELF_VEL_TAU = 0.08             # s smoothing of our own velocity (from successive pose estimates)
HEADING_MIN_SPEED = 40.0        # cm/s: slower than this, an enemy's direction of travel is mostly noise
HEADING_HOLD_S = 0.3            # s an enemy keeps its last heading after slowing down (longer = stale)
RADIO_DELAY = 0.03              # s assumed age of a heartbeat when it arrives (the two clocks differ)
MATE_STALE = 0.5                # s without a heartbeat before the teammate is dropped
MATE_EXTRAPOLATE_MAX = 0.3      # s at most to extrapolate the teammate along its velocity
SHARED_ENEMY_STALE = 0.3        # s a teammate's enemy report stays usable
SHARED_ENEMY_MERGE = 30.0       # cm: a reported enemy this close to one of ours is the same robot
SHARED_CONF_SCALE = 0.7         # confidence factor on enemies we only know from the teammate
FREE, ME, MATE, ENEMY = 0, 1, 2, 3   # occupancy codes

_cx = -L.WALL_X + CELL_W * (np.arange(COLS) + 0.5)
_cy = -L.WALL_Y + CELL_H * (np.arange(ROWS) + 0.5)
_GX, _GY = np.meshgrid(_cx, _cy)                      # ROWS x COLS cell centres


# =============================== grid ====================================
def cell_of(x, y) -> Tuple[int, int]:
    """(col, row) of the cell containing field point (x, y); clamped onto the grid."""
    col = min(max(int((x + L.WALL_X) // CELL_W), 0), COLS - 1)
    row = min(max(int((y + L.WALL_Y) // CELL_H), 0), ROWS - 1)
    return col, row


def cell_centre(col, row) -> Tuple[float, float]:
    """Field coordinates of a cell's centre."""
    return float(_cx[col]), float(_cy[row])


def heading_of(vx, vy) -> float:
    """Heading (0 = +y, CCW+) of a field-frame direction."""
    return math.atan2(-vx, vy)


# =============================== state ===================================
@dataclass
class RobotState:
    x: float                          # cm, field
    y: float
    heading: Optional[float]          # rad, 0 = facing the enemy goal, CCW+; None = unknown
    vx: float = 0.0                   # cm/s, field
    vy: float = 0.0
    confidence: float = 1.0           # 0..1
    age: float = 0.0                  # s since it was last actually measured
    id: int = 0                       # track id (enemies) / robot id (us, teammate)
    source: str = 'self'              # 'self', 'radio', 'lidar', 'teammate' (enemy only the teammate sees)

    @property
    def cell(self) -> Tuple[int, int]:
        return cell_of(self.x, self.y)

    @property
    def heading_deg(self) -> Optional[float]:
        return None if self.heading is None else math.degrees(L.wrap(self.heading))

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class WorldState:
    t: float
    valid: bool                       # False while we are lost: then nothing below is trustworthy
    me: RobotState
    mate: Optional[RobotState] = None
    enemies: List[RobotState] = field(default_factory=list)
    ball: Optional[Tuple[float, float]] = None
    tactics: dict = field(default_factory=dict)     # enemy_tracker.world_model_fields(...)
    mate_role: Optional[str] = None   # teammate's role from its heartbeat (None = no radio)
    mate_has_ball: bool = False
    mate_age: Optional[float] = None  # s since the teammate's last heartbeat

    # ---- geometry from our robot's point of view ----
    def to_robot(self, x, y) -> Tuple[float, float]:
        """Field point -> robot frame (right, forward) in cm."""
        dx, dy = x - self.me.x, y - self.me.y
        c, s = math.cos(self.me.heading), math.sin(self.me.heading)
        return c * dx + s * dy, -s * dx + c * dy

    def to_field(self, right, forward) -> Tuple[float, float]:
        """Robot frame (right, forward) cm -> field point."""
        c, s = math.cos(self.me.heading), math.sin(self.me.heading)
        return self.me.x + c * right - s * forward, self.me.y + s * right + c * forward

    def me_to(self, x, y) -> Tuple[float, float]:
        """(distance cm, bearing rad) from us to a field point; bearing in the robot frame,
        0 = straight ahead, CCW+ (positive = to our left)."""
        r, f = self.to_robot(x, y)
        return math.hypot(r, f), math.atan2(-r, f)

    def nearest_enemy(self) -> Optional[RobotState]:
        return min(self.enemies, key=lambda e: math.hypot(e.x - self.me.x, e.y - self.me.y), default=None)

    def occupancy(self, inflate=0.0, include_me=True) -> np.ndarray:
        """ROWS x COLS int8 grid: FREE, ME, MATE or ENEMY for every cell whose centre lies within
        ROBOT_R + inflate of that robot. grid[row, col]; row 0 = our goal end."""
        g = np.zeros((ROWS, COLS), np.int8)
        robots = ([(self.me, ME)] if include_me else []) + ([(self.mate, MATE)] if self.mate else [])
        robots += [(e, ENEMY) for e in self.enemies]
        for r, code in robots:
            g[np.hypot(_GX - r.x, _GY - r.y) <= ROBOT_R + inflate] = code
        return g

    def as_dict(self) -> dict:
        """Plain dict for logging / sending to a laptop."""
        rs = lambda r: None if r is None else {
            'x': round(r.x, 1), 'y': round(r.y, 1), 'cell': r.cell,
            'heading_deg': None if r.heading is None else round(r.heading_deg, 1),
            'vx': round(r.vx, 1), 'vy': round(r.vy, 1), 'conf': round(r.confidence, 2),
            'age': round(r.age, 3), 'id': r.id, 'source': r.source}
        return {'t': self.t, 'valid': self.valid, 'me': rs(self.me), 'mate': rs(self.mate),
                'enemies': [rs(e) for e in self.enemies], 'ball': self.ball,
                'tactics': self.tactics}


# ============================ radio heartbeat ============================
# 32 bytes, fits an nRF24 payload: magic, id+flags, our pose and velocity, up to 2 enemies, CRC8.
# flags byte: bits 0-3 robot id, 0x10 localised, 0x20 role is striker (else goalie),
# 0x40 role field valid (older senders leave it 0 = role unknown), 0x80 has the ball.
HB_MAGIC = 0xA5
_HB = struct.Struct('<BBhhhhhB' + 'hhhhB' * 2 + 'B')
HB_SIZE = _HB.size                    # 32


def _i16(v, scale):
    return int(max(-32768, min(32767, round(v * scale))))


def pack_heartbeat(robot_id, localised, me, enemies=(), role=None, has_ball=False):
    """me / enemies: RobotState (field frame). Positions in mm, speeds in mm/s, heading in 1e-4 rad.
    role: 'striker' / 'goalie' / None (armin_roles); has_ball: ball on our dribbler."""
    flags = (robot_id & 0x0F) | (0x10 if localised else 0) | (0x80 if has_ball else 0)
    if role is not None:
        flags |= 0x40 | (0x20 if role == 'striker' else 0)
    vals = [HB_MAGIC, flags, _i16(me.x, 10), _i16(me.y, 10), _i16(L.wrap(me.heading or 0.0), 1e4),
            _i16(me.vx, 10), _i16(me.vy, 10), min(len(enemies), 2)]
    for i in range(2):
        if i < len(enemies):
            e = enemies[i]
            vals += [_i16(e.x, 10), _i16(e.y, 10), _i16(e.vx, 10), _i16(e.vy, 10),
                     int(max(0, min(255, round(e.confidence * 255))))]
        else:
            vals += [0, 0, 0, 0, 0]
    body = _HB.pack(*vals, 0)[:-1]
    return body + bytes([L.crc8(body)])


def unpack_heartbeat(data):
    """-> (robot_id, localised, RobotState me, [RobotState enemies], role, has_ball) or None if
    corrupt. role is 'striker' / 'goalie', or None when the sender didn't say."""
    if len(data) != HB_SIZE or data[0] != HB_MAGIC or L.crc8(data[:-1]) != data[-1]:
        return None
    v = _HB.unpack(data)
    rid, loc_ok = v[1] & 0x0F, bool(v[1] & 0x10)
    role = ('striker' if v[1] & 0x20 else 'goalie') if v[1] & 0x40 else None
    me = RobotState(v[2] / 10, v[3] / 10, v[4] / 1e4, v[5] / 10, v[6] / 10, id=rid, source='radio')
    enemies = []
    for i in range(v[7]):
        b = 8 + 5 * i
        enemies.append(RobotState(v[b] / 10, v[b + 1] / 10, None, v[b + 2] / 10, v[b + 3] / 10,
                                  confidence=v[b + 4] / 255, source='teammate'))
    return rid, loc_ok, me, enemies, role, bool(v[1] & 0x80)


# =============================== model ===================================
class WorldModel:
    """Builds a WorldState from the Localiser, its EnemyTracker and the teammate's heartbeats."""

    def __init__(self, localiser, robot_id=1):
        self.loc = localiser
        self.robot_id = robot_id
        self.prev = None                   # (t, x, y) for our velocity
        self.vel = (0.0, 0.0)
        self.mate = None                   # (t_measured, localised, RobotState)
        self.mate_enemies = (None, [])     # (t_measured, [RobotState])
        self.mate_role = (None, None, False)   # (t_measured, role, has_ball)
        self.enemy_heading = {}            # track id -> (heading, t when last moving)
        self.state = None

    # ---- radio ----
    def on_heartbeat(self, data, t_rx):
        """A teammate's heartbeat arrived (bytes from the radio, our clock)."""
        hb = unpack_heartbeat(data)
        if hb is None:
            return False
        rid, ok, me, enemies, role, has_ball = hb
        t = t_rx - RADIO_DELAY
        self.mate = (t, ok, me)
        self.mate_role = (t, role, has_ball)
        if ok:
            self.loc.set_teammate(t, me.x, me.y)      # lets the tracker tell it from an enemy
            self.mate_enemies = (t, enemies)
        return True

    def heartbeat(self, role=None, has_ball=False):
        """Our heartbeat to send to the teammate (call after update())."""
        s = self.state
        if s is None:
            return pack_heartbeat(self.robot_id, False, RobotState(0, 0, 0), role=role, has_ball=has_ball)
        return pack_heartbeat(self.robot_id, s.valid, s.me, [e for e in s.enemies if e.source == 'lidar'],
                              role=role, has_ball=has_ball)

    # ---- the snapshot ----
    def update(self, t, ball=None):
        loc = self.loc
        x, y, th = loc.est
        # our velocity from successive lidar-corrected poses (not the wheels: they slip)
        if self.prev is not None and t > self.prev[0]:
            dt = t - self.prev[0]
            a = dt / (SELF_VEL_TAU + dt)
            vx = (x - self.prev[1]) / dt
            vy = (y - self.prev[2]) / dt
            self.vel = (self.vel[0] + a * (vx - self.vel[0]), self.vel[1] + a * (vy - self.vel[1]))
        self.prev = (t, x, y)
        valid = not loc.lost
        me = RobotState(x, y, L.wrap(th), self.vel[0], self.vel[1], loc.conf if valid else 0.0,
                        0.0, self.robot_id, 'self')

        mate = None
        if self.mate is not None and self.mate[1] and t - self.mate[0] < MATE_STALE:
            tm, _, m = self.mate
            age = t - tm
            k = min(age, MATE_EXTRAPOLATE_MAX)
            mate = RobotState(m.x + m.vx * k, m.y + m.vy * k, m.heading, m.vx, m.vy, 1.0, age, m.id, 'radio')

        enemies = []
        if valid:
            for e in loc.get_enemies(t):
                h = None
                if math.hypot(e.vx, e.vy) >= HEADING_MIN_SPEED:
                    h = heading_of(e.vx, e.vy)
                    self.enemy_heading[e.id] = (h, t)
                elif e.id in self.enemy_heading and t - self.enemy_heading[e.id][1] < HEADING_HOLD_S:
                    h = self.enemy_heading[e.id][0]
                enemies.append(RobotState(e.x, e.y, h, e.vx, e.vy, e.confidence, t - e.last_seen, e.id, 'lidar'))
            # an enemy only the teammate can see (ours blocked / outside the swept arc)
            tm, reported = self.mate_enemies
            if tm is not None and t - tm < SHARED_ENEMY_STALE:
                for r in reported:
                    if len(enemies) >= ET.MAX_ENEMIES:
                        break
                    age = t - tm
                    rx, ry = r.x + r.vx * age, r.y + r.vy * age
                    if all(math.hypot(rx - e.x, ry - e.y) > SHARED_ENEMY_MERGE for e in enemies):
                        h = heading_of(r.vx, r.vy) if math.hypot(r.vx, r.vy) >= HEADING_MIN_SPEED else None
                        enemies.append(RobotState(rx, ry, h, r.vx, r.vy, r.confidence * SHARED_CONF_SCALE,
                                                  age, 0, 'teammate'))
        self.enemy_heading = {k: v for k, v in self.enemy_heading.items() if t - v[1] < HEADING_HOLD_S}

        # tactics see every enemy we know of (ours + teammate-only) and the teammate as a blocker
        tactics = ET.world_model_fields((x, y, th), ball, enemies,
                                        mate or (loc.tracker.teammate(t) if valid else None), valid)
        tr, role, has_ball = self.mate_role
        fresh = tr is not None and t - tr < MATE_STALE
        self.state = WorldState(t, valid, me, mate, enemies, ball, tactics,
                                role if fresh else None, has_ball and fresh,
                                t - tr if fresh else None)
        return self.state
