"""
sim_world.py — the simulated world shared by sim_test.py (numbers) and field_viewer.py
(picture): robot shapes at scan height, ray-casting, D500 packet generation with noise,
junk returns and mixed pixels, and odometry with scale error / gyro drift.

Robots at scan height are a set of small circles in the robot frame (+y forward, +x right):
  'body'     a solid 18 cm cylinder — the lidar sees an arc of it
  'handles'  only the two carry handles on either side poke up into the scan plane
  'bits'     handles plus a mast and a bracket: a few small, scattered returns
With the scan plane above the 10 cm goals, 'handles' / 'bits' is what you usually get.
"""
import math
import random
import struct

import lidar_loc as L

ROBOT_R = 9.0
SHAPES = {
    'body': ((0.0, 0.0, 9.0),),
    'handles': ((-7.0, 0.0, 1.2), (7.0, 0.0, 1.2)),
    'bits': ((-7.0, 0.0, 1.2), (7.0, 0.0, 1.2), (0.0, 6.5, 1.5), (3.0, -6.0, 1.0)),
}
POSTS = ((-30.0, 109.5, 117.0), (30.0, 109.5, 117.0),       # goal side walls: x, y0, y1
         (-30.0, -117.0, -109.5), (30.0, -117.0, -109.5))
NOISE = 1.0          # cm range noise
P_JUNK = 0.02        # junk return probability
P_MIX = 0.25         # chance a ray at a depth edge returns a range in between (mixed pixel)
MIX_EDGE = 10.0      # cm: a range step this big between rays is a depth edge
HIT_WALL, HIT_MIXED, HIT_JUNK = -1, -2, -3


class Bot:
    """A robot following path(t) -> (x, y). heading(t) -> theta (0 faces +y, CCW+), or
    None to face the direction of travel. Set .fixed = (x, y) to pin it in place."""

    def __init__(self, name, kind, path, shape='handles', heading=None):
        self.name, self.kind, self.path, self.shape, self.heading = name, kind, path, shape, heading
        self.fixed = None
        self._last_th = 0.0

    def pos(self, t):
        return self.fixed if self.fixed is not None else self.path(t)

    def vel(self, t, h=1e-3):
        if self.fixed is not None:
            return 0.0, 0.0
        a, b = self.path(t + h), self.path(t - h)
        return (a[0] - b[0]) / (2 * h), (a[1] - b[1]) / (2 * h)

    def theta(self, t):
        if self.heading is not None:
            return self.heading(t)
        vx, vy = self.vel(t)
        if math.hypot(vx, vy) > 5:
            self._last_th = math.atan2(-vx, vy)
        return self._last_th

    def circles(self, t):
        x, y = self.pos(t)
        th = self.theta(t)
        c, s = math.cos(th), math.sin(th)
        return [(x + c * lx - s * ly, y + s * lx + c * ly, r) for lx, ly, r in SHAPES[self.shape]]


def ray_scene(x, y, dx, dy, circles, posts):
    """circles: [(cx, cy, r, owner)]. Returns (distance, owner or HIT_WALL)."""
    tx = ((L.WALL_X if dx > 0 else -L.WALL_X) - x) / dx if abs(dx) > 1e-9 else 1e9
    ty = ((L.WALL_Y if dy > 0 else -L.WALL_Y) - y) / dy if abs(dy) > 1e-9 else 1e9
    d = min(tx, ty)
    if ty < tx and abs(x + dx * ty) < 30: d += 7.4 / abs(dy)    # into the goal box
    hit = HIT_WALL
    for cx, cy, r, owner in circles:
        ox, oy = cx - x, cy - y
        c = ox * ox + oy * oy - r * r
        if c <= 0: continue                                     # lidar inside it: ignore
        b = ox * dx + oy * dy
        disc = b * b - c
        if disc > 0 and 0 < b - math.sqrt(disc) < d: d, hit = b - math.sqrt(disc), owner
    if posts and abs(dx) > 1e-9:
        for px, y0, y1 in POSTS:
            tp = (px - x) / dx
            if 0 < tp < d and y0 <= y + dy * tp <= y1: d, hit = tp, HIT_WALL
    return d, hit


class LidarSim:
    """Generates D500 packets for a moving robot in a scene, point by point at 5 kHz."""

    def __init__(self, seed=2, mixed=True, junk=True):
        self.rng = random.Random(seed)
        self.k = 0                    # packets generated
        self.prev = None              # previous ray's true distance (for mixed pixels)
        self.mixed, self.junk = mixed, junk

    def next_packet(self, us, bots, posts=False):
        """us(t) -> true (x, y, theta). Returns (t_rx, bytes, [(t, hit_x, hit_y, owner)])."""
        rng = self.rng
        ts = [(self.k * 12 + i) / 5000 for i in range(12)]
        self.k += 1
        angs, dists, hits = [], [], []
        for t in ts:
            x, y, th = us(t)
            ang = (3600 * t) % 360
            a = th - math.radians(ang)                          # CW lidar -> CCW robot -> field
            dx, dy = -math.sin(a), math.cos(a)
            circ = [(cx, cy, r, i) for i, b in enumerate(bots) for cx, cy, r in b.circles(t)]
            d, hit = ray_scene(x, y, dx, dy, circ, posts)
            true_d = d
            if self.mixed and self.prev is not None and abs(d - self.prev) > MIX_EDGE and rng.random() < P_MIX:
                lo, hi = sorted((d, self.prev))
                d, hit = rng.uniform(lo, hi), HIT_MIXED
            self.prev = true_d
            d += rng.gauss(0, NOISE)
            if self.junk and rng.random() < P_JUNK:
                d, hit = rng.uniform(5, 300), HIT_JUNK
            angs.append(ang)
            dists.append(int(max(d, 0) * 10))
            hits.append((t, x + dx * d, y + dy * d, hit))
        body = [0x54, 0x2C, 3600, int(angs[0] * 100)]
        for d in dists: body += [min(d, 65535), 200]
        body += [int(angs[-1] * 100), int(round(ts[-1] * 1000)) % 30000]
        raw = struct.pack('<BBHH' + 'HB' * 12 + 'HH', *body)
        raw += bytes([L.crc8(raw)])
        if rng.random() < 0.01: raw = raw[:20] + b'\x00' + raw[21:]   # corrupt packet
        return ts[-1] + L.PKT_TX_S + rng.uniform(0, 0.003), raw, hits


class Odometry:
    """Body-frame velocity with 5 % scale error and noise, and a yaw with slow gyro drift,
    as the robot's wheels / IMU would report them. Never sees kidnaps."""

    def __init__(self, seed=3):
        self.rng = random.Random(seed)
        self.yaw_bias = 0.0

    def sample(self, t, x, y, th, vx, vy, om, dt):
        c, s = math.cos(th), math.sin(th)
        vbx = (c * vx + s * vy) * 1.05 + self.rng.gauss(0, 3)
        vby = (-s * vx + c * vy) * 1.05 + self.rng.gauss(0, 3)
        self.yaw_bias += 0.01 * abs(om) * dt
        return vbx, vby, th + self.yaw_bias
