"""
lidar_loc.py — RCJ Soccer Open field localisation from an LDROBOT D500 (STL-19P)
                + BNO085 yaw + wheel (or optical-flow) odometry.

Why this exists: the D500 spins at ~10 Hz. At 3 m/s the robot moves 30 cm during one
revolution, and if it is spinning at 6 rad/s its heading changes ~34 deg. A "scan" is
therefore NOT a snapshot — every point was taken from a different pose. Treating it as a
snapshot smears the walls and the pose fit is wrong by tens of cm.

The fix, in one sentence: stamp every point with its own time, use IMU + odometry
(accurate over 25 ms even though they drift over seconds) to move every point into the
robot's frame at one reference instant, then fit that "unscrambled" batch to the known
wall rectangle with a robust, outlier-rejecting least-squares fit fused into a Kalman filter.

We also never wait for a full revolution: the lidar streams 5000 points/s, so we fit a
25 ms batch (~125 points, ~90 deg of arc) 40 times a second. The 10 Hz spin rate stops
mattering. A batch that only sees one wall corrects only that axis; the filter carries
the other axis on odometry until the next wall comes round.

Pipeline (each stage testable alone):
  D500Parser    raw UART bytes -> CRC-checked 47-byte packets
  LidarClock    lidar's own ms counter -> host monotonic time (no Linux UART jitter)
  packet_points packet -> per-point (time, angle, range, intensity)
  PoseBuffer    pure-odometry pose history, interpolated at each point's timestamp
  deskew        every point -> robot frame at the batch's reference instant
  point_mask    raw / self-body / tilt / edge (mixed-pixel) outlier filters
  fit_pose      Huber-weighted Gauss-Newton vs the wall rectangle, with the EKF prior
                as a term, so unobserved axes stay put; residual gate splits points
                into wall inliers / obstacles (robots) / impossible (beyond the wall)
  global_search kickoff + "I'm lost" relocalisation (grid search, IMU heading)
  Localiser     ties it together: on_odometry() at 100-200 Hz, on_lidar_bytes() as data
                arrives; read .est (x, y, theta), .P, .conf, .obstacles
  EnemyTracker  (enemy_tracker.py) fed each accepted batch's obstacle points; read
                .get_enemies() / .world_fields(ball) — empty while the Localiser is lost

Frames (same as perception/sensors-to-worldmodel.md):
  field: origin centre, +y toward enemy goal, +x right, cm, rad, theta=0 faces +y, CCW+
  robot: +y forward, +x right. field_point = (x, y) + R(theta) @ robot_point
"""
import math
import struct
from collections import deque

import numpy as np

import enemy_tracker

# =============================== CONFIG ==================================
# --- field (inside faces of the walls; verify against the event's field) ---
WALL_X, WALL_Y = 91.0, 121.5
# Goal regions are dropped from the fit when the scan plane is below goal height
# (10 cm) — the ray goes into the goal box and reads long. Above goal height with
# 22 cm walls you can set MASK_GOALS = False and use those wall points too.
MASK_GOALS = True
GOAL_MASKS = ((-36.0, 36.0, 104.0, 135.0), (-36.0, 36.0, -135.0, -104.0))  # xmin,xmax,ymin,ymax

# --- lidar mounting / protocol ---
LIDAR_XY = (0.0, 0.0)    # lidar spin centre in robot frame, cm
LIDAR_YAW = 0.0          # rad: robot-frame direction of the lidar's 0 deg mark (CCW+)
LIDAR_CW = True          # angles increase clockwise seen from above — VERIFY with a box
TS_WRAP = 30000          # packet timestamp wraps here (ms) — verify on your unit
BAUD = 230400
PKT_TX_S = 47 * 10 / BAUD   # time to clock one packet out of the UART (~2 ms)
LATENCY_TRIM = 0.0       # s, residual latency — tune with the spin test (see notes)
SELF_MASK_DEG = ()       # ((lo, hi), ...) lidar angles blocked by your own robot
                         # (mirror posts etc.). Record a scan in open space: < 12 cm = self.
SCAN_TO_WALL_TOP = 6.0   # cm between scan plane and the nearer of wall top / floor

# --- outlier rejection ---
MIN_R, MAX_R = 4.0, 320.0   # cm; field diagonal is ~305 cm
MIN_INTENSITY = 10
JUMP = 12.0                 # cm range jump to a neighbour => edge / mixed pixel
CORNER_EXCL = 8.0           # cm; points this close to two walls are ambiguous
MIN_INCIDENCE = 0.35        # cos(incidence); grazing hits are unreliable
HUBER = 2.0                 # cm
SIGMA_PT = 1.5              # cm, per-point noise for the fit
GATE_MIN, GATE_MAX = 6.0, 30.0

# --- filter ---
BATCH_S = 0.025             # lidar update every 25 ms
K_TRANS = 0.05              # odometry error: 5% of distance travelled (slip)
K_ROT = 0.02                # IMU yaw error: 2% of rotation
MIN_POINTS = 25
MIN_INLIER_FRAC = 0.40
LOST_AFTER = 8              # consecutive rejected batches (=200 ms) => relocalise
CHI2_REJECT = 25.0          # Mahalanobis^2 of a correction vs prior => reject
SELF_R = 9.0                # cm, OUR robot's radius: its centre can't be closer than this to a wall
K_SLIP = 1.0                # rejected batch => position sigma on each axis >= this x the odometry distance
                            # along that axis since walls last fixed it (wheels slipping on a wall / robot)
SLIP_SIG_MAX = 40.0         # cm: odometry unconfirmed for longer than this on an axis => relocalise
AXIS_MIN_PTS = 10           # wall inliers needed for a batch to count as having fixed that axis


# ============================ SE(2) helpers ==============================
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def between(a, b):
    """Motion from pose a to pose b, expressed in a's frame."""
    return compose(inverse(a), b)


# ============================ 1. Packet parser ===========================
def _crc_table(poly=0x4D):
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return bytes(t)


CRC_TABLE = _crc_table()


def crc8(data):
    c = 0
    for b in data:
        c = CRC_TABLE[c ^ b]
    return c


PKT_LEN = 47
_FMT = struct.Struct('<BBHH' + 'HB' * 12 + 'HHB')


class D500Parser:
    """Byte stream -> packets. Header 0x54 0x2C, 12 points, CRC8 (poly 0x4D)."""

    def __init__(self):
        self.buf = bytearray()
        self.bad_crc = 0
        self.good = 0

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(b'\x54\x2c')
            if i < 0:
                del self.buf[:-1]
                break
            if len(self.buf) - i < PKT_LEN:
                del self.buf[:i]
                break
            pkt = bytes(self.buf[i:i + PKT_LEN])
            if crc8(pkt[:-1]) != pkt[-1]:
                self.bad_crc += 1          # corrupt packet = outlier source #1
                del self.buf[:i + 1]
                continue
            del self.buf[:i + PKT_LEN]
            f = _FMT.unpack(pkt)
            self.good += 1
            out.append({'speed': f[2], 'start': f[3] / 100.0, 'dist': f[4:28:2],
                        'inten': f[5:28:2], 'end': f[28] / 100.0, 'ts': f[29]})
        return out


# ============================ 2. Clock sync ==============================
class LidarClock:
    """Map the lidar's ms counter onto host time.

    Host receive times jitter by milliseconds under Linux; the lidar's own counter
    doesn't. offset = running minimum of (rx_time - lidar_time), allowed to creep up
    slowly to follow crystal drift. The minimum is the packet that suffered no delay.
    """

    def __init__(self, drift=2e-4):
        self.last = None
        self.base = 0
        self.off = None
        self.tl_prev = None
        self.drift = drift

    def to_host(self, ts_ms, t_rx):
        if self.last is not None and ts_ms < self.last - TS_WRAP // 2:
            self.base += TS_WRAP
        self.last = ts_ms
        tl = (self.base + ts_ms) * 1e-3
        o = (t_rx - PKT_TX_S) - tl
        if self.off is None or abs(o - self.off) > 1.0:
            self.off = o                       # first packet, or resync after a gap
        else:
            self.off = min(o, self.off + self.drift * (tl - self.tl_prev))
        self.tl_prev = tl
        return tl + self.off + LATENCY_TRIM


def packet_points(p, t_pkt):
    """Per-point time/angle. Points are evenly spaced between start and end angle."""
    n = 12
    span = (p['end'] - p['start']) % 360.0
    step = span / (n - 1)
    k = np.arange(n)
    ang = (p['start'] + step * k) % 360.0
    dt = step / max(p['speed'], 1.0)          # speed field is deg/s
    t = t_pkt - dt * (n - 1 - k)               # timestamp taken as the LAST point
    return (t, ang, np.asarray(p['dist'], float) / 10.0,
            np.asarray(p['inten'], float))


# ============================ 3. Pose history ============================
class PoseBuffer:
    def __init__(self, n=1024):
        self.n = n
        self.t = np.zeros(n)
        self.p = np.zeros((n, 3))
        self.i = 0

    def push(self, t, pose):
        j = self.i % self.n
        self.t[j] = t
        self.p[j] = pose
        self.i += 1

    def latest_t(self):
        return self.t[(self.i - 1) % self.n] if self.i else -math.inf

    def at(self, tq):
        k = min(self.i, self.n)
        idx = np.arange(self.i - k, self.i) % self.n
        t, p = self.t[idx], self.p[idx]
        th = np.unwrap(p[:, 2])
        tq = np.clip(tq, t[0], t[-1])
        return np.interp(tq, t, p[:, 0]), np.interp(tq, t, p[:, 1]), np.interp(tq, t, th)


# ============================ 4. Unscramble ==============================
def lidar_to_robot(ang_deg, r):
    a = LIDAR_YAW + np.radians(-ang_deg if LIDAR_CW else ang_deg)
    return LIDAR_XY[0] - r * np.sin(a), LIDAR_XY[1] + r * np.cos(a)


def deskew(odom, t, qx, qy, t_ref):
    """Move each point (robot frame at its own time t_i) into the robot frame at t_ref."""
    xi, yi, thi = odom.at(t)
    xr, yr, thr = odom.at(np.array([t_ref]))
    c, s = np.cos(thi), np.sin(thi)
    dx = xi + c * qx - s * qy - xr
    dy = yi + s * qx + c * qy - yr
    cr, sr = np.cos(thr), np.sin(thr)
    return cr * dx + sr * dy, -sr * dx + cr * dy


def point_mask(ang, r, inten, tilt, edges=True):
    """Sensor-level rejection — cheap, before any geometry. edges=False skips the
    mixed-pixel filter (the enemy tracker needs those points: a robot's handle is a
    1-2 point blip with a big jump on both sides, and does its own, gentler test)."""
    ok = (r > MIN_R) & (r < MAX_R) & (inten >= MIN_INTENSITY)
    for lo, hi in SELF_MASK_DEG:                       # own chassis / mirror posts
        ok &= ~((ang >= lo) & (ang <= hi))
    # tilt: a pitched robot's scan plane hits the floor or clears the wall at range
    ok &= r * math.tan(tilt) < SCAN_TO_WALL_TOP
    if not edges:
        return ok
    # mixed pixels: at a depth edge the beam straddles two surfaces and returns a range
    # in between (a "veil" of ghost points). Drop anything next to a big jump.
    rr = np.where(ok, r, np.nan)
    jump = np.zeros_like(r, bool)
    d = np.abs(np.diff(rr))
    big = ~(d < JUMP)                                  # nan counts as a jump
    jump[1:] |= big
    jump[:-1] |= big
    return ok & ~jump


# ============================ 5. Robust fit ==============================
def _goal_masked(Px, Py):
    if not MASK_GOALS:
        return np.zeros_like(Px, bool)
    m = np.zeros_like(Px, bool)
    for x0, x1, y0, y1 in GOAL_MASKS:
        m |= (Px > x0) & (Px < x1) & (Py > y0) & (Py < y1)
    return m


def fit_pose(qx, qy, prior, P_prior, iters=4):
    """MAP fit: sum Huber(wall residual)/sigma^2 + (pose - prior)' P^-1 (pose - prior).

    Returns pose, covariance, and per-point classification from the final iteration.
    The prior term is what makes a one-wall batch safe: the unconstrained direction
    has no data, so it simply stays at the prior.
    """
    x = np.array(prior, float)
    Pinv = np.linalg.inv(P_prior)
    sig = math.sqrt(max(P_prior[0, 0], P_prior[1, 1]))
    gate = min(max(3 * sig + 4.0, GATE_MIN), GATE_MAX)
    for _ in range(iters):
        c, s = math.cos(x[2]), math.sin(x[2])
        Px = x[0] + c * qx - s * qy
        Py = x[1] + s * qx + c * qy
        rx = np.abs(Px) - WALL_X                       # signed: negative = inside field
        ry = np.abs(Py) - WALL_Y
        on_x = np.abs(rx) < np.abs(ry)
        r = np.where(on_x, rx, ry)
        corner = (np.abs(rx) < CORNER_EXCL) & (np.abs(ry) < CORNER_EXCL)
        dist = np.hypot(Px - x[0], Py - x[1]) + 1e-6
        inc = np.where(on_x, np.abs(Px - x[0]), np.abs(Py - x[1])) / dist
        inlier = (np.abs(r) < gate) & ~corner & (inc > MIN_INCIDENCE) & ~_goal_masked(Px, Py)
        sx, sy = np.sign(Px), np.sign(Py)
        J = np.zeros((len(r), 3))
        J[:, 0] = np.where(on_x, sx, 0.0)
        J[:, 1] = np.where(on_x, 0.0, sy)
        J[:, 2] = np.where(on_x, -sx * (Py - x[1]), sy * (Px - x[0]))
        w = np.where(np.abs(r) <= HUBER, 1.0, HUBER / np.maximum(np.abs(r), 1e-9))
        w = w * inc ** 2 * inlier / SIGMA_PT ** 2
        H = (J * w[:, None]).T @ J + Pinv
        g = (J * w[:, None]).T @ r + Pinv @ (x - prior)
        dx = -np.linalg.solve(H, g)
        x += dx
        if abs(dx[0]) + abs(dx[1]) < 0.05 and abs(dx[2]) < 1e-4:
            break
    cls = np.where(inlier, 0, np.where(r < -gate, 1, 2))   # 0 wall, 1 obstacle, 2 impossible
    return x, np.linalg.inv(H), cls, Px, Py, int(on_x[inlier].sum()), int((~on_x[inlier]).sum())


def global_search(qx, qy, thetas, step=4.0):
    """Grid search over position for each candidate heading. Score = wall hits minus
    points that would lie outside the field. ~20 ms in numpy for ~400 points."""
    xs = np.arange(-WALL_X + 10, WALL_X - 9, step)
    ys = np.arange(-WALL_Y + 10, WALL_Y - 9, step)
    best = (-1e9, None)
    for th in thetas:
        c, s = math.cos(th), math.sin(th)
        wx = c * qx - s * qy
        wy = s * qx + c * qy
        Px = xs[:, None, None] + wx[None, None, :]
        Py = ys[None, :, None] + wy[None, None, :]
        d = np.minimum(np.abs(np.abs(Px) - WALL_X), np.abs(np.abs(Py) - WALL_Y))
        outside = (np.abs(Px) > WALL_X + 4) | (np.abs(Py) > WALL_Y + 4)
        score = (d < 4.0).sum(-1) - 2 * outside.sum(-1)
        k = np.unravel_index(np.argmax(score), score.shape)
        if score[k] > best[0]:
            best = (float(score[k]), (xs[k[0]], ys[k[1]], th))
    return best[1], best[0] / max(len(qx), 1)


# ============================ 6. Localiser ===============================
def _inside(pose):
    """Clamp a pose so the robot's centre stays SELF_R inside the walls."""
    mx, my = WALL_X - SELF_R, WALL_Y - SELF_R
    return (min(max(pose[0], -mx), mx), min(max(pose[1], -my), my), pose[2])


class Localiser:
    def __init__(self, pose=(0.0, -60.0, 0.0), heading_trusted=True):
        """heading_trusted: IMU heading already referenced to the field (latched at
        kickoff, or from the camera's goal colours). If False, relocalisation also
        tries theta + pi — the rectangle is 180-deg symmetric, so lidar alone cannot
        tell the two ends apart."""
        self.est = tuple(pose)
        self.P = np.diag([25.0 ** 2, 25.0 ** 2, 0.1 ** 2])
        self.odom = (0.0, 0.0, 0.0)
        self.odom_buf = PoseBuffer()
        self.t_odom = None
        self.yaw_prev = None
        self.tilt = 0.0
        self.heading_trusted = heading_trusted
        self.parser = D500Parser()
        self.clock = LidarClock()
        self.pend = []
        self.recent = deque(maxlen=5)     # ~125 ms of raw points for relocalisation
        self.bad = 0
        self.lost = True
        self.conf = 0.0
        self.obstacles = np.empty((0, 2))
        self.stats = {}
        self.moved = [0.0, 0.0]           # odometry |dx|, |dy| (field) since walls last fixed each axis
        self.tracker = enemy_tracker.EnemyTracker()   # separate from localisation; fed per batch

    # ---- called from the IMU/odometry loop, 100-200 Hz ----
    def on_odometry(self, t, vx_b, vy_b, yaw, roll=0.0, pitch=0.0):
        """vx_b, vy_b: body velocity cm/s (x right, y forward) from wheel kinematics or
        an optical-flow sensor. yaw: BNO085 game-rotation-vector yaw (rad, CCW+); only
        its change is used, so its zero doesn't matter."""
        self.tilt = math.hypot(roll, pitch)
        if self.t_odom is None:
            self.t_odom, self.yaw_prev = t, yaw
            self.odom_buf.push(t, self.odom)
            return
        dt = t - self.t_odom
        if dt <= 0:
            return
        dth = wrap(yaw - self.yaw_prev)
        thm = self.odom[2] + 0.5 * dth
        c, s = math.cos(thm), math.sin(thm)
        new = (self.odom[0] + (c * vx_b - s * vy_b) * dt,
               self.odom[1] + (s * vx_b + c * vy_b) * dt,
               self.odom[2] + dth)
        rel = between(self.odom, new)
        cand = compose(self.est, rel)
        moved = _inside(cand)                        # pushing a wall: wheels turn, robot can't go through it
        if moved[:2] != cand[:2]:
            # ...and neither may the motion history that deskewing replays, or every batch
            # taken while pinned to a wall gets smeared by motion that never happened.
            rel = between(self.est, moved)
            new = compose(self.odom, rel)
        self.moved[0] += abs(moved[0] - self.est[0])  # only motion the walls allowed counts as slip risk
        self.moved[1] += abs(moved[1] - self.est[1])
        self.est = moved
        self.odom = new
        self.odom_buf.push(t, new)
        self.t_odom, self.yaw_prev = t, yaw
        step = math.hypot(vx_b, vy_b) * dt
        q = np.diag([(K_TRANS * step + 0.02) ** 2, (K_TRANS * step + 0.02) ** 2,
                     (K_ROT * abs(dth) + 2e-4) ** 2])
        self.P = self.P + q

    # ---- called whenever UART bytes arrive ----
    def on_lidar_bytes(self, data, t_rx):
        for p in self.parser.feed(data):
            t_pkt = self.clock.to_host(p['ts'], t_rx)
            self.pend.append(packet_points(p, t_pkt))
            if self.pend[-1][0][-1] - self.pend[0][0][0] >= BATCH_S:
                self._batch()

    def _batch(self):
        t, ang, r, inten = (np.concatenate(z) for z in zip(*self.pend))
        self.pend = []
        ok = point_mask(ang, r, inten, self.tilt)
        if t[-1] > self.odom_buf.latest_t() + 0.05:
            return                                       # odometry stalled: don't guess
        raw = (t, ang, r, inten)                         # every point, for the enemy tracker
        qx, qy = lidar_to_robot(ang[ok], r[ok])
        t = t[ok]
        if len(t) < MIN_POINTS:
            return
        self.recent.append((t, qx, qy))
        t_ref = float(t[-1])
        dqx, dqy = deskew(self.odom_buf, t, qx, qy, t_ref)

        ox, oy, oth = self.odom_buf.at(np.array([t_ref]))
        rel = between((ox[0], oy[0], oth[0]), self.odom)  # motion since t_ref
        prior = compose(self.est, inverse(rel))

        if self.lost:
            self._relocalise(t_ref, prior)
            return

        pose, cov, cls, Px, Py, nx, ny = fit_pose(dqx, dqy, np.array(prior), self.P)
        frac = (cls == 0).mean()
        d = pose - np.array(prior)
        d[2] = wrap(d[2])
        m2 = float(d @ np.linalg.solve(self.P, d))
        self.stats = {'n': len(t), 'inlier': frac, 'x_wall': nx, 'y_wall': ny, 'm2': m2}
        if frac >= MIN_INLIER_FRAC and m2 < CHI2_REJECT:
            self.est = _inside(compose(tuple(pose), rel))
            self.P = cov
            self.bad = 0
            if nx >= AXIS_MIN_PTS:                # only an axis the walls actually pinned is fixed
                self.moved[0] = 0.0
            if ny >= AXIS_MIN_PTS:
                self.moved[1] = 0.0
        else:
            self.bad += 1
            # The usual reason a good batch disagrees is the wheels, not the lidar: slipping
            # against a wall or a robot, odometry runs away at full speed and a tight P then
            # throws out every wall point that would correct it. The error on an axis can't
            # exceed the odometry motion along it since walls last fixed that axis, so let
            # sigma cover that: the next batch that sees those walls pulls the pose back.
            for i in (0, 1):
                g = K_SLIP * self.moved[i]
                self.P[i, i] = max(self.P[i, i], g * g)
            if self.bad >= LOST_AFTER or max(self.moved) * K_SLIP > SLIP_SIG_MAX:
                self.lost = True                  # too far to pull back: global search
        sig = math.sqrt(self.P[0, 0] + self.P[1, 1])
        self.conf = float(np.clip(frac, 0, 1) * np.clip(1 - sig / 20.0, 0, 1))
        self.obstacles = np.column_stack([Px[cls == 1], Py[cls == 1]])
        self._track(t_ref, frac >= MIN_INLIER_FRAC and m2 < CHI2_REJECT, pose, cov, raw)

    # ---- enemy tracking (reads the localisation result, never feeds back into it) ----
    def _track(self, t_ref, accepted, pose, cov, raw):
        if self.lost:
            self.tracker.reset()               # field-frame enemies are meaningless now
            return
        if not accepted:
            self.tracker.coast(t_ref)          # don't place enemies with a pose we just rejected
            return
        t, ang, r, inten = raw
        ok = point_mask(ang, r, inten, self.tilt, edges=False)   # keep handle blips
        qx, qy = lidar_to_robot(ang[ok], r[ok])
        dqx, dqy = deskew(self.odom_buf, t[ok], qx, qy, t_ref)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        fx = pose[0] + c * dqx - s * dqy       # whole batch in field frame at the fitted pose
        fy = pose[1] + s * dqx + c * dqy
        lidar = (pose[0] + c * LIDAR_XY[0] - s * LIDAR_XY[1],
                 pose[1] + s * LIDAR_XY[0] + c * LIDAR_XY[1])
        self.tracker.update(t_ref, fx, fy, t[ok], lidar, cov)

    def set_teammate(self, t, x, y):
        """Teammate field pose from the radio heartbeat (t = when it was measured)."""
        self.tracker.set_teammate(t, x, y)

    def get_enemies(self, t=None):
        """Confirmed enemies predicted to t (default: latest odometry time). Empty while
        lost — enemy positions are field coordinates, so a bad self-pose makes them all wrong."""
        if self.lost:
            return []
        return self.tracker.enemies(self.t_odom if t is None else t)

    def world_fields(self, ball, t=None):
        """WorldModel obstacle / opponent fields for the current pose and a field-frame ball."""
        t = self.t_odom if t is None else t
        mate = None if self.lost else self.tracker.teammate(t)
        return enemy_tracker.world_model_fields(self.est, ball, self.get_enemies(t), mate,
                                                valid=not self.lost)

    def _relocalise(self, t_ref, prior):
        t, qx, qy = (np.concatenate(z) for z in zip(*self.recent))
        dqx, dqy = deskew(self.odom_buf, t, qx, qy, t_ref)
        if len(t) < 150:                                 # want most of a revolution
            return
        th = prior[2]
        ths = [th + e for e in (-0.06, 0.0, 0.06)]
        if not self.heading_trusted:
            ths += [th + math.pi + e for e in (-0.06, 0.0, 0.06)]
        guess, score = global_search(dqx, dqy, ths)
        if guess is None or score < 0.5:
            return
        P0 = np.diag([6.0 ** 2, 6.0 ** 2, 0.08 ** 2])
        pose, cov, cls, *_ = fit_pose(dqx, dqy, np.array(guess), P0, iters=8)
        if (cls == 0).mean() < 0.6:
            return
        ox, oy, oth = self.odom_buf.at(np.array([t_ref]))
        rel = between((ox[0], oy[0], oth[0]), self.odom)
        self.est = _inside(compose(tuple(pose), rel))
        self.P = cov
        self.lost = False
        self.bad = 0
        self.moved = [0.0, 0.0]
