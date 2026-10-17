"""
enemy_tracker.py — turn the Localiser's lidar points into tracked enemy robots.

Input, once per 25 ms lidar batch (from lidar_loc.Localiser._batch): EVERY valid point of
the batch (field frame, scan order, with timestamps) — including the depth-edge points the
localiser's own mixed-pixel filter throws away — plus the lidar's field position at the
batch reference time and the self-pose covariance.

Why every point: with the scan plane above the 10 cm goals, a robot is usually NOT a solid
body at scan height. The lidar sees its handles on either side, a mast, a bit of bumper —
one or two points each, with a big range jump to the wall on both sides. A "drop anything
next to a jump" filter deletes all of that. Here a point is only dropped as a mixed pixel
when its range lies BETWEEN its two neighbours (a ghost smeared across an edge); a point
NEARER than both neighbours is a thin object and is kept.

Pipeline (each stage testable alone):
  obstacle_points  inside the walls by a margin, mixed pixels removed, goal side walls
                   removed point by point (only when the scan is below goal height)
  cluster_points   scan-order segments (angular + euclidean gaps), grouped into robots:
                   consecutive segments that fit inside one robot's footprint, never mixing
                   segments that belong to different existing tracks. Centre per group:
                     'arc'      a real body arc (enough points, wide enough, fits a circle of
                                radius ROBOT_R) -> known-radius circle fit, tight covariance
                     'fragment' handles / small bits -> centre of the region of all positions
                                within ROBOT_R of every visible point (two handles -> their
                                midpoint), covariance = spread of that region
  suppress         wall artefacts, teammate flagging (radio pose), clamp into the field
  EnemyTracker     constant-velocity Kalman filter per track, gating (Mahalanobis on the
                   centre + every point inside the predicted footprint), brute-force global
                   nearest neighbour, tentative -> confirmed, visibility-aware misses
                   (a 25 ms batch sees ~90 deg: tracks outside it or occluded COAST; a miss
                   needs rays straight through everywhere the robot could be, so rays
                   passing between a robot's handles never count)
  world_model_fields  WorldModel inputs from the tracks + robot pose + ball

Each cluster is timestamped with the mean time of its points, not t_ref: a batch spans
25 ms, and a robot at 2 m/s moves 5 cm in that time. Deskewing only removes OUR motion.

Frames: same as lidar_loc — field origin at centre, +y toward the enemy goal, +x right,
cm, rad, theta = 0 faces +y, CCW positive. Robot frame: +y forward, +x right.
"""
import math
from dataclasses import dataclass

import numpy as np

# =============================== CONFIG ==================================
# --- field / geometry ---
WALL_X, WALL_Y = 91.0, 121.5    # inside faces of the walls, cm (keep equal to lidar_loc)
ROBOT_R = 9.0                   # robot footprint radius, cm (RCJ Open: 18 cm max diameter)
BALL_R = 3.7                    # ball radius, cm (RCJ 74 mm ball)
SCAN_ABOVE_GOALS = True         # scan plane above the 10 cm goals (Singapore Open setup): goal structure is invisible, skip that filter
GOAL_SEGMENTS = (               # goal side walls (x0, y0, x1, y1) that show up as obstacles when the scan is below goal height
    (-30.0, 109.5, -30.0, 117.0), (30.0, 109.5, 30.0, 117.0),
    (-30.0, -109.5, -30.0, -117.0), (30.0, -109.5, 30.0, -117.0),
)
GOAL_STRUCT_TOL = 3.0           # cm: a point this close to a goal side wall is goal structure, not robot
WALL_MARGIN = 5.0               # cm: drop clusters whose centre is closer than this to a wall (corner / mixed-pixel artefacts)
GOAL_MOUTH_X = 30.0             # enemy goal mouth half-width, cm (posts at |x| = 30)
GOAL_MOUTH_Y = 109.5            # y of the enemy goal mouth (front face of the posts), cm

# --- obstacle points ---
OBST_WALL_MARGIN = 6.0          # cm: a point must be this far inside the walls to be an obstacle (wall noise)
MIX_JUMP = 10.0                 # cm: a point whose range lies between its neighbours', each this far off, is a mixed pixel

# --- clustering ---
ANG_GAP_DEG = 4.0               # split segments where consecutive points are this far apart in bearing (lidar step ~0.72 deg)
EUCLID_GAP = 6.0                # cm: split segments where consecutive points are this far apart
GROUP_MAX_SPAN = 20.0           # cm: segments are one robot only if all of them fit within this span (18 cm robot + noise)
GROUP_LOOKBACK = 3              # a segment may join any of the last this-many groups (skips a junk point inside an arc)
OWN_PAD_MAX = 6.0               # cm: cap on the track-uncertainty pad used to tie segments to a confirmed track
MIN_CLUSTER_PTS = 2             # points a group needs to be used at all (a lone point could be junk, or any part of a robot)
ARC_MIN_PTS = 3                 # points needed to treat a group as a body arc
ARC_MIN_CHORD = 7.0             # cm: a body arc must be at least this wide (a handle is ~2 cm and must NOT be circle-fitted)
ARC_MIN_FILL = 0.7              # a body arc returns at least this fraction of the rays across its angular width (two handles don't)
ARC_STEP_DEG = 0.72             # deg between lidar rays (5000 pts/s at 10 rev/s)
ARC_MAX_RMS = 1.5               # cm: circle-fit residual above this -> not a body arc, use the fragment estimate
CIRCLE_ITERS = 2                # Gauss-Newton iterations of the known-radius circle fit
CIRCLE_TANG_PRIOR = 0.25        # weight pulling the fitted centre's bearing to the arc centroid's (1/cm^2, vs 1/cm^2 per point)
FEAS_GRID = 13                  # fragment centre: candidate grid points per axis ...
FEAS_HALF = 15.0                # cm ... over +/- this around the points' centroid (room for a stray point pulling it)
FEAS_TOL = 1.5                  # cm: slack on "every point within ROBOT_R of the centre" (noise, handle thickness)
FRAG_COV_SCALE = 1.5            # x fragment covariance: the visible bits needn't be symmetric about the centre

# --- measurement noise (per cluster centre) ---
MEAS_SIG_BASE = 0.6             # cm radial sigma of an arc fit at zero range with MEAS_N_REF points
MEAS_SIG_PER_M = 0.3            # cm of extra radial sigma per metre of range
MEAS_N_REF = 12                 # point count at which the sigmas above apply; scales as sqrt(MEAS_N_REF / n)
MEAS_TANG_FACTOR = 1.6          # arc tangential sigma = this x radial (a visible arc pins range better than bearing)
MEAS_SIG_MIN = 0.5              # cm floor on an arc sigma
FRAG_SIG_MIN = 1.5              # cm floor on a fragment sigma
EDGE_DEG = 2.0                  # deg: a cluster this close to either end of the swept arc is clipped by the batch boundary
EDGE_TANG_INFLATE = 3.0         # x tangential sigma for a clipped cluster (only part of the robot seen)

# --- teammate ---
TEAMMATE_GATE = 22.0            # cm: clusters within this of the radio teammate pose are the teammate
TEAMMATE_MAX_AGE = 0.3          # s: ignore radio teammate poses older than this
TEAMMATE_PURGE_S = 0.3          # s: delete an enemy track that has sat on the teammate pose this long

# --- tracker ---
ACCEL_SIGMA = 600.0             # cm/s^2 white-acceleration process noise (3 m/s robots, hard stops and turns)
MAX_SPEED = 350.0               # cm/s: velocity cap so one bad association can't launch a phantom
INIT_VEL_SIGMA = 150.0          # cm/s velocity sigma of a newborn track
MAX_POS_SIGMA = 30.0            # cm cap on coasting position sigma, so a blind track can't gate half the field
COAST_DAMP_AFTER = 0.15         # s without a hit before coasting velocity starts to decay
COAST_VEL_TAU = 0.4             # s decay time constant of coasting velocity (robots don't keep going blind)
GATE_CHI2 = 13.8                # Mahalanobis^2 gate on the centre (99.9 %, 2 dof)
GATE_PAD = 2.0                  # cm: every point of an associated cluster lies within ROBOT_R + 3 sigma + this of the track
TENTATIVE_PENALTY = 2.0         # extra assignment cost for tentative tracks, so confirmed tracks win ties
BIRTH_MIN_SEP = 25.0            # cm: no new track this close to an existing one (avoids duplicates)
CONFIRM_HITS = 2                # hits needed to confirm a track (a robot is scanned ~once per 100 ms revolution)
CONFIRM_WINDOW = 0.15           # s: ... within this long of its first hit, else the tentative track dies
DELETE_AFTER = 0.5              # s without hits, with at least one genuine miss, before a track is deleted
MAX_COAST = 1.5                 # s without any hit (occluded / never in the arc) before a track is deleted anyway
MAX_ENEMIES = 2                 # confirmed enemy tracks exposed (2v2)
MAX_TRACKS = 6                  # hard cap on tracks incl. tentative (keeps brute-force assignment small)
HIT_SCORE_TAU = 0.3             # s decay of the hit score that picks the best MAX_ENEMIES tracks (recent hits win)
MISS_MIN_RAYS = 3               # rays across a track's footprint needed before an empty footprint counts as a miss
VIS_MARGIN = 6.0                # cm slack on the line-of-sight check (ray ended in front = occluded, beyond = passed through)

# --- world model ---
ENEMY_NEAR_BALL = 30.0          # cm centre-to-ball distance for enemy_near_ball
N_LANES = 5                     # shot lanes sampled across the enemy goal mouth
SHOT_MIN_GAP = 6.0              # cm of clear goal mouth (for the ball centre) to call a shot lane clear
BLOCK_CONF = 0.3                # an enemy blocks lanes / goal span only above this confidence


# ============================== helpers ==================================
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# Small index helpers (np.r_ costs ~15 us a call; these are ~1 us).
def _starts(brk):
    """Segment start indices given a boolean 'break after element i' array."""
    return np.concatenate(([0], np.flatnonzero(brk) + 1))


def _ends(starts, n):
    return np.concatenate((starts[1:], [n]))


_g = np.linspace(-FEAS_HALF, FEAS_HALF, FEAS_GRID)
_GRID = np.stack(np.meshgrid(_g, _g), -1).reshape(-1, 2)   # candidate centre offsets
_GRID_VAR = (_g[1] - _g[0]) ** 2 / 12                         # grid quantisation variance


@dataclass
class Cluster:
    x: float
    y: float
    t: float                    # mean time of its points
    n: int
    nseg: int                   # segments grouped into it (2 handles = 2)
    width: float                # cm, span of its points
    range: float
    cov: np.ndarray             # 2x2 measurement covariance, cm^2
    mode: str                   # 'arc' (body circle fit) or 'fragment' (handles / bits)
    qx: np.ndarray              # its points, field frame
    qy: np.ndarray
    clipped: bool = False       # touches an end of the swept arc (only part of the robot seen)
    teammate: bool = False


@dataclass
class Enemy:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    cov: np.ndarray             # 4x4 over (x, y, vx, vy)
    confidence: float
    age: float                  # s since the track was born
    last_seen: float            # host time of the last hit
    coasting: bool              # no hit in the last revolution


@dataclass
class Teammate:
    x: float
    y: float
    t: float
    source: str                 # 'lidar' (matched cluster) or 'radio'


@dataclass
class Track:
    id: int
    x: np.ndarray               # (x, y, vx, vy) at time t
    P: np.ndarray
    t: float
    first_t: float
    last_hit: float
    hits: int = 1
    score: float = 1.0
    misses: int = 0             # genuine misses since the last hit
    confirmed: bool = False
    mate_since: float = None    # host time it started sitting on the teammate pose


# ============================ 1. Obstacle points =========================
def _on_goal_structure(qx, qy):
    """Points lying on a goal side wall (only visible when the scan is below goal height)."""
    on = np.zeros(len(qx), bool)
    for x0, y0, x1, y1 in GOAL_SEGMENTS:
        sx, sy = x1 - x0, y1 - y0
        s = np.clip(((qx - x0) * sx + (qy - y0) * sy) / (sx * sx + sy * sy), 0.0, 1.0)
        on |= np.hypot(qx - (x0 + s * sx), qy - (y0 + s * sy)) < GOAL_STRUCT_TOL
    return on


def obstacle_points(fx, fy, rr):
    """Mask of points that may belong to a robot: well inside the walls, not a mixed pixel,
    and (scan below goal height) not on a goal side wall. Goal structure is removed point
    by point, BEFORE grouping, so a goalie standing against a post keeps its own points and
    is never thrown away with the post. rr: ranges from the lidar, scan order."""
    obst = (np.abs(fx) < WALL_X - OBST_WALL_MARGIN) & (np.abs(fy) < WALL_Y - OBST_WALL_MARGIN)
    if not SCAN_ABOVE_GOALS:
        obst &= ~_on_goal_structure(fx, fy)
    mixed = np.zeros(len(rr), bool)
    if len(rr) >= 3:
        dp = rr[1:-1] - rr[:-2]                 # vs previous point
        dn = rr[1:-1] - rr[2:]                  # vs next point
        mixed[1:-1] = (np.abs(dp) > MIX_JUMP) & (np.abs(dn) > MIX_JUMP) & (dp * dn < 0)
    return obst & ~mixed, mixed


# ============================ 2. Clustering ==============================
def _group_segments(px, py, s, e, owners):
    """Greedy scan-order grouping of segments into robots. A segment joins one of the last
    GROUP_LOOKBACK groups (most recent first) if everything still fits within GROUP_MAX_SPAN
    and the two are not tied to different existing tracks. Looking back past the latest
    group lets a robot's arc or handles re-unite across a stray junk point between them.
    Pure Python: there are only a handful of segments. Returns [[segments], points, owner]."""
    xs, ys = px.tolist(), py.tolist()
    groups = []
    for i in range(len(s)):
        a, b = int(s[i]), int(e[i]) - 1
        ends = [(xs[a], ys[a]), (xs[b], ys[b])]
        if b - a > 1:
            m = (a + b) // 2
            ends.append((xs[m], ys[m]))
        own, best = None, None
        for k, (tx, ty, pad) in enumerate(owners):
            dmax = max(math.hypot(x - tx, y - ty) for x, y in ends)
            if dmax <= ROBOT_R + pad and (best is None or dmax < best):
                own, best = k, dmax
        for g in reversed(groups[-GROUP_LOOKBACK:]):
            if g[2] is not None and own is not None and g[2] != own:
                continue
            if max(math.hypot(p[0] - q[0], p[1] - q[1]) for p in ends for q in g[1]) <= GROUP_MAX_SPAN:
                g[0].append(i)
                g[1] += ends
                if g[2] is None:
                    g[2] = own
                break
        else:
            groups.append([[i], ends, own])
    return groups


def _circle_fit(qx, qy, first, cnt, lx, ly):
    """Known-radius circle fit for contiguous groups (vectorised). Returns centre, unit
    line-of-sight vectors, rms residual and a success mask."""
    k = len(first)
    lab = np.repeat(np.arange(k), cnt)
    cx, cy = np.add.reduceat(qx, first) / cnt, np.add.reduceat(qy, first) / cnt
    ux, uy = cx - lx, cy - ly
    un = np.hypot(ux, uy) + 1e-9
    ux, uy = ux / un, uy / un
    tx, ty = -uy, ux
    fx0, fy0 = cx + ROBOT_R * ux, cy + ROBOT_R * uy
    c_x, c_y = fx0.copy(), fy0.copy()
    lam = CIRCLE_TANG_PRIOR
    lxx, lxy, lyy = lam * tx * tx, lam * tx * ty, lam * ty * ty
    A = np.empty((5, len(lab)))
    for _ in range(CIRCLE_ITERS):
        dx, dy = qx - c_x[lab], qy - c_y[lab]
        d = np.hypot(dx, dy) + 1e-9
        jx, jy = dx / -d, dy / -d
        r = d - ROBOT_R
        np.multiply(jx, jx, out=A[0])
        np.multiply(jx, jy, out=A[1])
        np.multiply(jy, jy, out=A[2])
        np.multiply(jx, r, out=A[3])
        np.multiply(jy, r, out=A[4])
        hxx, hxy, hyy, gx, gy = np.add.reduceat(A, first, axis=1)
        tp = lam * (tx * (c_x - fx0) + ty * (c_y - fy0))
        hxx, hxy, hyy = hxx + lxx, hxy + lxy, hyy + lyy
        gx, gy = gx + tx * tp, gy + ty * tp
        det = hxx * hyy - hxy * hxy + 1e-9
        c_x = c_x - (hyy * gx - hxy * gy) / det
        c_y = c_y - (hxx * gy - hxy * gx) / det
    res = np.hypot(qx - c_x[lab], qy - c_y[lab]) - ROBOT_R
    rms = np.sqrt(np.add.reduceat(res * res, first) / cnt)
    ok = ((rms < ARC_MAX_RMS) & (np.hypot(c_x - fx0, c_y - fy0) < ROBOT_R)
          & ((c_x - cx) * ux + (c_y - cy) * uy > 0) & np.isfinite(c_x))
    return c_x, c_y, ux, uy, ok


def _fragment_fit(qx, qy, first, cnt, lx, ly):
    """Centre of the feasible region {c : |p - c| <= ROBOT_R + FEAS_TOL for (nearly) every point p}
    (vectorised over groups), its covariance, and the line-of-sight unit vectors. Two handles
    -> a thin lens around their midpoint; one blob -> a disc around it. No guess about which
    side is hidden: a lone handle is usually the one that didn't fall between two rays."""
    k = len(first)
    lab = np.repeat(np.arange(k), cnt)
    cx, cy = np.add.reduceat(qx, first) / cnt, np.add.reduceat(qy, first) / cnt
    ux, uy = cx - lx, cy - ly
    un = np.hypot(ux, uy) + 1e-9
    ux, uy = ux / un, uy / un
    gx = cx[:, None] + _GRID[None, :, 0]                       # k x G candidate centres
    gy = cy[:, None] + _GRID[None, :, 1]
    d = np.hypot(qx[:, None] - gx[lab], qy[:, None] - gy[lab])  # n x G
    # Consensus, not unanimity: the centres within reach of the MOST points, so one stray
    # junk / mixed point grouped in with a robot can't drag the estimate.
    votes = np.add.reduceat((d <= ROBOT_R + FEAS_TOL).astype(np.int16), first, axis=0)  # k x G
    w = (votes == votes.max(1, keepdims=True)).astype(float)
    nw = w.sum(1)
    mx, my = (w * gx).sum(1) / nw, (w * gy).sum(1) / nw
    vxx = (w * (gx - mx[:, None]) ** 2).sum(1) / nw + _GRID_VAR
    vyy = (w * (gy - my[:, None]) ** 2).sum(1) / nw + _GRID_VAR
    vxy = (w * (gx - mx[:, None]) * (gy - my[:, None])).sum(1) / nw
    k2 = FRAG_COV_SCALE
    return mx, my, k2 * vxx, k2 * vxy, k2 * vyy, ux, uy


def cluster_points(px, py, pt, brg, lidar_xy, pose_cov=None, arc=None, owners=()):
    """Group scan-ordered obstacle points (times pt, bearings brg from the lidar) into robots.

    arc: (bearing of the first, bearing of the last ray of the batch), field frame.
    owners: [(x, y, pad)] predicted centres of existing tracks, used so two neighbouring
    robots' fragments are never merged. Returns a list of Cluster.
    """
    n = len(px)
    if n == 0:
        return []
    lx, ly = lidar_xy
    split = ((np.abs(_wrap(np.diff(brg))) > math.radians(ANG_GAP_DEG))
             | (np.hypot(np.diff(px), np.diff(py)) > EUCLID_GAP))
    s = _starts(split)
    e = _ends(s, n)
    groups = _group_segments(px, py, s, e, owners)
    span = np.array([max(math.hypot(p[0] - q[0], p[1] - q[1]) for p in g[1] for q in g[1]) for g in groups])
    npts = np.array([sum(e[j] - s[j] for j in g[0]) for g in groups])
    keep = np.flatnonzero((span <= GROUP_MAX_SPAN)     # one segment wider than a robot: not a robot
                          & (npts >= MIN_CLUSTER_PTS))
    if len(keep) == 0:
        return []
    # Gather each kept group's points together (scan order inside a group is preserved).
    idx = np.concatenate([np.arange(s[j], e[j]) for gi in keep for j in groups[gi][0]])
    cnt = np.array([sum(e[j] - s[j] for j in groups[gi][0]) for gi in keep])
    nseg = np.array([len(groups[gi][0]) for gi in keep])
    span = span[keep]
    k = len(keep)
    qx, qy, qt, qb = px[idx], py[idx], np.asarray(pt, float)[idx], brg[idx]
    first = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    last = first + cnt - 1

    # Fragment estimate for every group; body arcs then get the (much tighter) circle fit.
    mx, my, vxx, vxy, vyy, ux, uy = _fragment_fit(qx, qy, first, cnt, lx, ly)
    mode = np.zeros(k, bool)                                    # True = arc
    # Body arc = enough points, wide enough, and DENSE across its angle (a split arc still
    # qualifies; two handles 14 cm apart with wall between them don't).
    angw = np.abs(_wrap(qb[last] - qb[first]))
    fill = cnt / (angw / math.radians(ARC_STEP_DEG) + 1)
    cand = np.flatnonzero((cnt >= ARC_MIN_PTS) & (span >= ARC_MIN_CHORD) & (fill >= ARC_MIN_FILL))
    if len(cand):
        sel = np.concatenate([np.arange(first[i], last[i] + 1) for i in cand])
        cf = np.concatenate(([0], np.cumsum(cnt[cand])[:-1]))
        c_x, c_y, aux, auy, ok = _circle_fit(qx[sel], qy[sel], cf, cnt[cand], lx, ly)
        ci = cand[ok]
        mode[ci] = True
        mx[ci], my[ci] = c_x[ok], c_y[ok]
        ux[ci], uy[ci] = aux[ok], auy[ok]
    tx, ty = -uy, ux
    rng = np.hypot(mx - lx, my - ly)

    clipped = np.zeros(k, bool)
    if arc is not None:
        edge = math.radians(EDGE_DEG)
        clipped = ((np.abs(_wrap(qb[first] - arc[0])) < edge) | (np.abs(_wrap(qb[last] - arc[1])) < edge))
    infl = np.where(clipped, EDGE_TANG_INFLATE ** 2, 1.0)

    # Arc covariance: radial / tangential sigmas from range and point count.
    sr = np.maximum((MEAS_SIG_BASE + MEAS_SIG_PER_M * rng / 100.0) * np.sqrt(MEAS_N_REF / cnt), MEAS_SIG_MIN)
    sr2 = sr ** 2
    st2 = np.maximum(sr * MEAS_TANG_FACTOR, MEAS_SIG_MIN) ** 2 * infl
    axx = sr2 * ux * ux + st2 * tx * tx
    axy = sr2 * ux * uy + st2 * tx * ty
    ayy = sr2 * uy * uy + st2 * ty * ty
    # Fragment covariance: spread of the feasible region, floored, sideways-inflated if clipped.
    f2 = FRAG_SIG_MIN ** 2
    fxx = vxx + f2 + (infl - 1) * f2 * tx * tx
    fxy = vxy + (infl - 1) * f2 * tx * ty
    fyy = vyy + f2 + (infl - 1) * f2 * ty * ty
    cxx, cxy, cyy = np.where(mode, axx, fxx), np.where(mode, axy, fxy), np.where(mode, ayy, fyy)
    if pose_cov is not None:                      # self-pose: position, and heading error x range
        pc = np.asarray(pose_cov)
        th2 = rng ** 2 * float(pc[2, 2])
        cxx = cxx + pc[0, 0] + th2 * tx * tx
        cxy = cxy + pc[0, 1] + th2 * tx * ty
        cyy = cyy + pc[1, 1] + th2 * ty * ty
    ct = np.add.reduceat(qt, first) / cnt
    return [Cluster(float(mx[i]), float(my[i]), float(ct[i]), int(cnt[i]), int(nseg[i]),
                    float(span[i]), float(rng[i]),
                    np.array([[cxx[i], cxy[i]], [cxy[i], cyy[i]]]),
                    'arc' if mode[i] else 'fragment',
                    qx[first[i]:last[i] + 1], qy[first[i]:last[i] + 1], bool(clipped[i]))
            for i in range(k)]


# ======================= 3. False-positive suppression ===================
def suppress(clusters, mate_xy=None, enemy_xy=()):
    """Drop wall artefacts, flag the teammate, clamp centres (goal structure is already
    gone, point by point, in obstacle_points). A cluster within TEAMMATE_GATE of the radio teammate pose is flagged as the teammate,
    unless an existing enemy track is closer to it than the teammate is (an enemy duelling
    with our teammate must not vanish with it).
    """
    out = []
    for c in clusters:
        if abs(c.x) > WALL_X - WALL_MARGIN or abs(c.y) > WALL_Y - WALL_MARGIN:
            continue
        c.x = min(max(c.x, -(WALL_X - ROBOT_R)), WALL_X - ROBOT_R)
        c.y = min(max(c.y, -(WALL_Y - ROBOT_R)), WALL_Y - ROBOT_R)
        if mate_xy is not None:
            dm = math.hypot(c.x - mate_xy[0], c.y - mate_xy[1])
            if dm < TEAMMATE_GATE and all(math.hypot(c.x - ex, c.y - ey) >= dm for ex, ey in enemy_xy):
                c.teammate = True
        out.append(c)
    return out


# ============================ 4. Tracker =================================
def _assign(cost, ok, miss):
    """Brute-force global nearest neighbour. cost/ok: T x M lists, miss: T list.
    Returns, for each track, a cluster index or -1."""
    T, M = len(cost), (len(cost[0]) if cost else 0)
    best_cost, best = [math.inf], [[-1] * T]
    cur = [-1] * T
    used = [False] * M

    def rec(i, acc):
        if acc >= best_cost[0]:
            return
        if i == T:
            best_cost[0], best[0] = acc, cur.copy()
            return
        for j in range(M):
            if ok[i][j] and not used[j]:
                used[j] = True
                cur[i] = j
                rec(i + 1, acc + cost[i][j])
                used[j] = False
        cur[i] = -1
        rec(i + 1, acc + miss[i])

    rec(0, 0.0)
    return best[0]


_F = np.eye(4)


def _predict(x, P, dt, coasting_for):
    """Constant-velocity prediction (returns new arrays). Velocity decays once coasting."""
    if dt <= 0:
        return x.copy(), P.copy()
    F = _F.copy()
    F[0, 2] = F[1, 3] = dt
    q = ACCEL_SIGMA ** 2
    a, b, c = q * dt ** 4 / 4, q * dt ** 3 / 2, q * dt ** 2
    P = F @ P @ F.T + np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
    x = F @ x
    if coasting_for > COAST_DAMP_AFTER:
        f = math.exp(-dt / COAST_VEL_TAU)
        g = np.array([1.0, 1.0, f, f])
        x = x * g
        P = P * np.outer(g, g)
    sx, sy = math.sqrt(P[0, 0]), math.sqrt(P[1, 1])
    if sx > MAX_POS_SIGMA or sy > MAX_POS_SIGMA:                # cap position sigma
        g = np.array([min(1.0, MAX_POS_SIGMA / sx), min(1.0, MAX_POS_SIGMA / sy), 1.0, 1.0])
        P = P * np.outer(g, g)
    return x, P


class EnemyTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.t = None
        self.mate = None                 # (t, x, y) radio pose
        self.teammate_obs = None         # Teammate from the last flagged cluster
        self.clusters = []               # last batch's clusters (after suppression)
        self.last = None                 # last batch's points + masks, for viewers / debugging
        self.created = 0                 # tracks ever born (diagnostics)

    def reset(self):
        """Drop everything — call when self-localisation is lost (every enemy is then wrong)."""
        self.tracks = []
        self.t = None
        self.clusters = []
        self.teammate_obs = None

    def set_teammate(self, t, x, y):
        """Teammate field pose from the radio heartbeat (t = when it was measured)."""
        self.mate = (t, x, y)

    def _mate_xy(self, t):
        if self.mate is None or t - self.mate[0] > TEAMMATE_MAX_AGE:
            return None
        return self.mate[1], self.mate[2]

    # ---- main entry, once per lidar batch ----
    def update(self, t, fx, fy, ft, lidar_xy, pose_cov=None):
        """t: batch reference time. fx, fy, ft: EVERY valid point of the batch (field frame,
        scan order, edge points included) and its time. lidar_xy: lidar field position at t.
        Empty arrays = no measurement, just age the tracks."""
        fx, fy = np.asarray(fx, float), np.asarray(fy, float)
        ft = np.asarray(ft, float)
        lx, ly = lidar_xy
        mate = self._mate_xy(t)
        rb = rr = np.empty(0)
        cl = []
        pred = [_predict(tr.x, tr.P, t - tr.t, t - tr.last_hit) for tr in self.tracks]   # at batch time
        if len(fx):
            ex, ey = fx - lx, fy - ly
            rr, rb = np.hypot(ex, ey), np.arctan2(ey, ex)
            use, mixed = obstacle_points(fx, fy, rr)
            owners = [(x[0], x[1], min(3 * math.sqrt(max(P[0, 0], P[1, 1])), OWN_PAD_MAX) + GATE_PAD)
                      for tr, (x, P) in zip(self.tracks, pred) if tr.confirmed]   # confirmed tracks claim nearby segments
            i = np.flatnonzero(use)
            cl = cluster_points(fx[i], fy[i], ft[i], rb[i], lidar_xy, pose_cov, (rb[0], rb[-1]), owners)
            self.last = {'t': t, 'fx': fx, 'fy': fy, 'use': use, 'mixed': mixed, 'lidar': (lx, ly)}
        cl = suppress(cl, mate, [(tr.x[0], tr.x[1]) for tr in self.tracks if tr.confirmed])
        self.clusters = cl
        meas = []
        for c in cl:
            if c.teammate:
                self.teammate_obs = Teammate(c.x, c.y, c.t, 'lidar')
            else:
                meas.append(c)

        # --- association at each cluster's own time ---
        # Gate: Mahalanobis on the centre AND every point inside the predicted footprint
        # (a stray single point can't drag a track), then brute-force GNN.
        T, M = len(self.tracks), len(meas)
        assign = [-1] * T
        if T and M:
            # Tracks predicted (damped, capped) to the batch time, then nudged to each
            # cluster's own time (within the 25 ms batch).
            X = np.array([x for x, _ in pred])
            Z = np.array([[c.x, c.y] for c in meas])
            dt = np.array([c.t for c in meas])[None] - t                    # 1 x M, <= 0
            PX = X[:, None, :2] + X[:, None, 2:] * dt[..., None]             # T x M predicted centres
            D = Z[None] - PX
            Pp = np.array([P for _, P in pred])
            pxx = np.repeat(Pp[:, 0, 0][:, None], M, 1)
            pyy = np.repeat(Pp[:, 1, 1][:, None], M, 1)
            pxy = np.repeat(Pp[:, 0, 1][:, None], M, 1)
            R = np.array([c.cov for c in meas])
            sxx, sxy, syy = pxx + R[None, :, 0, 0], pxy + R[None, :, 0, 1], pyy + R[None, :, 1, 1]
            det = sxx * syy - sxy * sxy
            d2 = (syy * D[..., 0] ** 2 - 2 * sxy * D[..., 0] * D[..., 1] + sxx * D[..., 1] ** 2) / det
            # footprint check: farthest point of each cluster from each predicted centre
            Qx = np.concatenate([c.qx for c in meas])
            Qy = np.concatenate([c.qy for c in meas])
            qf = np.concatenate(([0], np.cumsum([len(c.qx) for c in meas])[:-1]))
            lab = np.repeat(np.arange(M), [len(c.qx) for c in meas])
            far = np.maximum.reduceat(np.hypot(Qx[None] - PX[:, lab, 0], Qy[None] - PX[:, lab, 1]), qf, axis=1)
            reach = ROBOT_R + 3 * np.sqrt(np.maximum(pxx, pyy)) + GATE_PAD
            pen = [0.0 if tr.confirmed else TENTATIVE_PENALTY for tr in self.tracks]
            cost = (d2 + np.array(pen)[:, None]).tolist()
            ok = ((d2 < GATE_CHI2) & (far <= reach)).tolist()
            assign = _assign(cost, ok, [GATE_CHI2 + p for p in pen])

        # --- Kalman updates at the cluster time ---
        used = set()
        for i, tr in enumerate(self.tracks):
            j = assign[i]
            if j < 0:
                continue
            c = meas[j]
            used.add(j)
            x, P = _predict(tr.x, tr.P, c.t - tr.t, c.t - tr.last_hit)
            S = P[:2, :2] + c.cov
            K = P[:, :2] @ np.linalg.inv(S)
            x = x + K @ (np.array([c.x, c.y]) - x[:2])
            P = P - K @ P[:2, :]
            P = 0.5 * (P + P.T)
            sp = math.hypot(x[2], x[3])
            if sp > MAX_SPEED:
                x[2:] *= MAX_SPEED / sp
            tr.x, tr.P, tr.t = x, P, max(c.t, tr.t)
            tr.score = tr.score * math.exp(-(c.t - tr.last_hit) / HIT_SCORE_TAU) + 1.0
            tr.hits += 1
            tr.last_hit = c.t
            tr.misses = 0
            if not tr.confirmed and tr.hits >= CONFIRM_HITS and c.t - tr.first_t <= CONFIRM_WINDOW:
                tr.confirmed = True

        # --- unassigned tracks: a miss only if rays crossed the whole predicted footprint
        # and every one of them went past it. Rays passing between a robot's handles are
        # normal, so a footprint that is only partly crossed, or with any return near the
        # predicted range, just coasts; so does a track outside the arc or behind something.
        un = [(tr, pred[i]) for i, tr in enumerate(self.tracks) if assign[i] < 0]
        if un and len(rb):
            for tr, (x, P) in un:
                reff = ROBOT_R + 2 * math.sqrt(max(P[0, 0], P[1, 1]))   # footprint + where it might be
                rho = math.hypot(x[0] - lx, x[1] - ly)
                if rho <= reff:
                    continue
                half = math.asin(min(1.0, reff / rho))
                off = _wrap(rb - math.atan2(x[1] - ly, x[0] - lx))
                win = np.abs(off) < half
                if win.sum() < MISS_MIN_RAYS or off[win].min() > -0.5 * half or off[win].max() < 0.5 * half:
                    continue                                       # footprint not crossed: coast
                r = rr[win]
                if (r < rho - reff - VIS_MARGIN).any():
                    continue                                       # something nearer blocks it: coast
                if (r > rho + reff + VIS_MARGIN).all():
                    tr.misses += 1                                 # rays went straight through it

        # --- births ---
        for j, c in enumerate(meas):
            if j in used or len(self.tracks) >= MAX_TRACKS:
                continue
            if any(math.hypot(c.x - tr.x[0], c.y - tr.x[1]) < BIRTH_MIN_SEP for tr in self.tracks):
                continue
            P = np.zeros((4, 4))
            P[:2, :2] = c.cov
            P[2, 2] = P[3, 3] = INIT_VEL_SIGMA ** 2
            self.tracks.append(Track(self.next_id, np.array([c.x, c.y, 0.0, 0.0]), P, c.t, c.t, c.t))
            self.next_id += 1
            self.created += 1

        # --- lifecycle ---
        keep = []
        for tr in self.tracks:
            since = t - tr.last_hit
            if not tr.confirmed and (since > CONFIRM_WINDOW or tr.misses > 0):
                continue
            if since > MAX_COAST or (since > DELETE_AFTER and tr.misses > 0):
                continue
            if mate is not None and math.hypot(tr.x[0] - mate[0], tr.x[1] - mate[1]) < TEAMMATE_GATE:
                if tr.mate_since is None:
                    tr.mate_since = t
                elif t - tr.mate_since > TEAMMATE_PURGE_S:
                    continue
            else:
                tr.mate_since = None
            keep.append(tr)
        conf = sorted((tr for tr in keep if tr.confirmed), key=lambda tr: -self._score(tr, t))
        drop = {id(tr) for tr in conf[MAX_ENEMIES:]}
        self.tracks = [tr for tr in keep if id(tr) not in drop]
        self.t = t

    def coast(self, t):
        """Batch rejected by the localiser: no measurements, just age the tracks."""
        self.update(t, (), (), (), (0.0, 0.0))

    @staticmethod
    def _score(tr, t):
        return tr.score * math.exp(-(t - tr.last_hit) / HIT_SCORE_TAU)

    # ---- outputs ----
    def enemies(self, t=None):
        """Confirmed enemy tracks, best first, predicted to time t (default: last batch)."""
        tq = self.t if t is None else t
        if tq is None:
            return []
        out = []
        for tr in sorted((tr for tr in self.tracks if tr.confirmed), key=lambda tr: -self._score(tr, tq)):
            x, P = _predict(tr.x, tr.P, tq - tr.t, tq - tr.last_hit)
            since = tq - tr.last_hit
            sig = math.sqrt(max(P[0, 0] + P[1, 1], 0.0))
            conf = (min(tr.hits, 10) / 10.0) ** 0.5 * math.exp(-max(0.0, since - 0.12) / 0.4) / (1 + sig / 10.0)
            out.append(Enemy(tr.id, float(x[0]), float(x[1]), float(x[2]), float(x[3]), P,
                             float(min(max(conf, 0.0), 1.0)), tq - tr.first_t, tr.last_hit, since > 0.12))
        return out

    def teammate(self, t):
        """Best teammate estimate: a fresh lidar match beats the (laggier) radio pose."""
        if self.teammate_obs is not None and t - self.teammate_obs.t < 0.15:
            return self.teammate_obs
        if self.mate is not None and t - self.mate[0] < TEAMMATE_MAX_AGE:
            return Teammate(self.mate[1], self.mate[2], self.mate[0], 'radio')
        return None


# ============================ 5. World model =============================
def _robot_bearing(pose, x, y):
    """Bearing of a field point from the robot, robot frame, 0 = forward, CCW+ (rad)."""
    dx, dy = x - pose[0], y - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    return math.atan2(-rx, ry)


def world_model_fields(pose, ball, enemies, teammate=None, valid=True):
    """WorldModel fields from the tracker output.

    pose: robot (x, y, theta) field frame. ball: (x, y) field frame or None.
    enemies: list of Enemy. teammate: Teammate or None. valid: False when the Localiser is
    lost — every field-frame enemy is then wrong, so nothing is reported.

    Definitions (check against the perception doc):
      obstacles            [(x, y, radius, kind, confidence)], kind 'enemy' / 'teammate'
      nearest_opp_range    cm, robot centre to the nearest enemy centre (None if no enemy)
      nearest_opp_bearing  rad, robot frame, 0 = forward, CCW+
      enemy_near_ball      any enemy centre within ENEMY_NEAR_BALL of the ball
      enemy_ball_side      side of the robot->ball line the enemy nearest the ball is on:
                           +1 left, -1 right, 0 none within 2 x ENEMY_NEAR_BALL
      lane_occupancy       N_LANES values in [0, 1] across the enemy goal mouth (-x to +x):
                           max confidence of anything blocking the ball -> lane line
      shot_lane_clear      some gap in the goal mouth >= SHOT_MIN_GAP is unblocked
      open_goal_span       (x0, x1) widest unblocked goal-mouth interval for the ball centre,
                           field x at y = GOAL_MOUTH_Y; None if fully blocked / no ball
    """
    f = {'valid': bool(valid), 'obstacles': [], 'nearest_opp_range': None,
         'nearest_opp_bearing': None, 'enemy_near_ball': False, 'enemy_ball_side': 0,
         'lane_occupancy': [0.0] * N_LANES, 'shot_lane_clear': False, 'open_goal_span': None}
    if not valid:
        return f
    blockers = [(e.x, e.y, e.confidence, 'enemy') for e in enemies]
    if teammate is not None:
        blockers.append((teammate.x, teammate.y, 1.0, 'teammate'))
    f['obstacles'] = [(x, y, ROBOT_R, kind, c) for x, y, c, kind in blockers]
    if enemies:
        d = [math.hypot(e.x - pose[0], e.y - pose[1]) for e in enemies]
        i = int(np.argmin(d))
        f['nearest_opp_range'] = d[i]
        f['nearest_opp_bearing'] = _robot_bearing(pose, enemies[i].x, enemies[i].y)
    if ball is None:
        return f
    bx, by = ball
    if enemies:
        db = [math.hypot(e.x - bx, e.y - by) for e in enemies]
        i = int(np.argmin(db))
        f['enemy_near_ball'] = db[i] < ENEMY_NEAR_BALL
        if db[i] < 2 * ENEMY_NEAR_BALL:
            cross = (bx - pose[0]) * (enemies[i].y - by) - (by - pose[1]) * (enemies[i].x - bx)
            f['enemy_ball_side'] = 1 if cross > 0 else -1

    # Shadows of the blockers on the goal mouth line, seen from the ball.
    h = GOAL_MOUTH_Y - by
    if h <= 0:
        return f
    lo, hi = -GOAL_MOUTH_X + BALL_R, GOAL_MOUTH_X - BALL_R
    rad = ROBOT_R + BALL_R
    shadows = []
    for x, y, c, kind in blockers:
        ex, ey = x - bx, y - by
        d = math.hypot(ex, ey)
        if ey < -rad or ey > h + rad or (kind == 'enemy' and c < BLOCK_CONF):
            continue
        if d <= rad:
            shadows.append((-math.inf, math.inf, c))               # touching the ball
            continue
        phi, a = math.atan2(ex, ey), math.asin(rad / d)
        p0, p1 = max(phi - a, -math.pi / 2 + 1e-6), min(phi + a, math.pi / 2 - 1e-6)
        if p0 < p1:
            shadows.append((bx + h * math.tan(p0), bx + h * math.tan(p1), c))
    lanes = np.linspace(lo, hi, N_LANES)
    occ = np.zeros(N_LANES)
    for s0, s1, c in shadows:
        occ = np.maximum(occ, np.where((lanes >= s0) & (lanes <= s1), c, 0.0))
    f['lane_occupancy'] = occ.tolist()
    free = [(lo, hi)]                                              # mouth minus the shadows
    for s0, s1, _c in shadows:
        nxt = []
        for a, b in free:
            if s1 <= a or s0 >= b:
                nxt.append((a, b))
                continue
            if s0 > a:
                nxt.append((a, s0))
            if s1 < b:
                nxt.append((s1, b))
        free = nxt
    if free:
        best = max(free, key=lambda ab: ab[1] - ab[0])
        if best[1] > best[0]:
            f['open_goal_span'] = best
            f['shot_lane_clear'] = best[1] - best[0] >= SHOT_MIN_GAP
    return f
