"""
field_viewer.py — drive a simulated robot and watch the REAL localiser and enemy tracker.

    python field_viewer.py          (numpy only; tkinter comes with Python)

The lidar bytes come from sim_world.py (same simulator as sim_test.py) and go through the
real lidar_loc.Localiser and enemy_tracker.EnemyTracker — nothing here is a re-implementation.

What you see
  solid            the TRUE world: walls, goals, our robot (white), enemies (red, 18 cm
                   footprint with their handles / bits), teammate (green)
  dots             lidar hits from the last revolution, at their TRUE positions, coloured by
                   what they hit (wall grey, enemy yellow, teammate pale green, mixed pixel
                   purple, junk blue)
  dashed           what the ROBOT BELIEVES, drawn relative to the real robot: the field
                   outline where it thinks the walls are (cyan), and each tracked enemy as an
                   18 cm circle with id, velocity arrow and 2-sigma ellipse (pink; fainter
                   while coasting). Perfect localisation = dashed field on top of the real one.
  thin white line  true enemy -> its track, labelled with the error the WorldModel would get

Controls
  W A S D   drive (field-relative: W = toward the enemy goal at the top)   Shift = fast
  Q / E     rotate left / right          R  robot-relative driving on/off
  T         auto-spin 6 rad/s on/off (stress test for deskewing)
  K         kidnap: teleport the robot (odometry doesn't see it) to test relocalisation
  1 / 2     enemies + teammate moving / standing still
  mouse     drag any robot to place it (it then stands still)
  H         robot shape at scan height: handles -> handles + bits -> full body
  G         scan below goal height (goal side walls visible, goal filter on) on/off
  P         points: true hits / what the tracker used / both
  C         field grid + occupancy (what WorldState.occupancy() gives strategy code)
  X / J     mixed pixels / junk returns on/off
  , / .     slower / faster (0.1x .. 1x)     Space  pause      Esc  quit
"""
import bisect
import collections
import math
import random
import time
import tkinter as tk

import numpy as np

import enemy_tracker as ET
import field_model as F
import lidar_loc as L
import sim_world as W
from sim_test import chaser, goalie, mate

# ------------------------------------------------------------------ look
S = 3.1                                   # px per cm (set from the screen height in main())
PAD = 14.0                                # cm drawn beyond the walls (goal boxes)
FW, FH = L.WALL_X + PAD, L.WALL_Y + PAD
CW, CH = int(2 * FW * S), int(2 * FH * S)
PANEL = 480
BG, FIELD, WALL = '#11161c', '#1d3b2a', '#e8e8e8'
C_US, C_EST = '#ffffff', '#38d6ff'
C_ENEMY, C_TRACK = '#ff4d4d', '#ff9ec7'
C_MATE, C_ERR = '#4ddc6b', '#ffffff'
C_HIT_ENEMY, C_HIT_MATE = '#ffe066', '#b6f5c3'      # lidar hits ON a robot
HIT_COL = {W.HIT_WALL: '#8a96a3', W.HIT_MIXED: '#d873ff', W.HIT_JUNK: '#6b5cff'}
SHAPES = ('handles', 'bits', 'body')

DRIVE, FAST, TURN = 120.0, 300.0, 4.0     # cm/s, cm/s, rad/s
ACCEL = 900.0                             # cm/s^2 robot acceleration limit
ODO_HZ = 200


def to_px(x, y):
    return (x + FW) * S, (FH - y) * S


def to_field(px, py):
    return px / S - FW, FH - py / S


class Viewer:
    def __init__(self, root):
        self.root = root
        root.title('RCJ lidar viewer — real Localiser + EnemyTracker')
        root.configure(bg=BG)
        self.cv = tk.Canvas(root, width=CW, height=CH, bg=BG, highlightthickness=0)
        self.cv.grid(row=0, column=0)
        self.hud = tk.Canvas(root, width=PANEL, height=CH, bg=BG, highlightthickness=0)
        self.hud.grid(row=0, column=1, sticky='n')
        self._draw_static()

        self.keys = {}                    # keysym -> pressed counter (debounced release)
        self.shape_i = 0
        self.robot_rel = False
        self.autospin = False
        self.paused = False
        self.speed = 1.0
        self.show_pts = 0                 # 0 true hits, 1 tracker view, 2 both
        self.show_grid = False
        self.posts = False
        self.drag = None
        for ev, fn in (('<KeyPress>', self.on_key), ('<KeyRelease>', self.on_key_up),
                       ('<ButtonPress-1>', self.on_press), ('<B1-Motion>', self.on_motion),
                       ('<ButtonRelease-1>', self.on_release)):
            root.bind(ev, fn)
        self.reset()
        self.last_wall = time.perf_counter()
        self.root.after(16, self.frame)

    # ------------------------------------------------------------ setup
    def reset(self):
        self.t = 0.0
        self.pose = [0.0, -60.0, 0.0]     # true x, y, theta
        self.vel = [0.0, 0.0, 0.0]        # true field vx, vy, omega
        self.hist_t, self.hist_p = [0.0], [tuple(self.pose)]
        self.loc = L.Localiser(pose=(10.0, -50.0, 0.0))
        self.lidar = W.LidarSim(seed=random.randrange(1 << 30))
        self.odo = W.Odometry(seed=random.randrange(1 << 30))
        self.radio = collections.deque()  # teammate heartbeats in flight: (arrival time, bytes)
        self.next_radio = 0.0
        self.wm = F.WorldModel(self.loc, robot_id=1)
        self.state = self.wm.update(0.0)
        self.hits = collections.deque()   # (t, x, y, owner) of the last revolution
        self.slipping = False
        self.tracker_ms = collections.deque(maxlen=80)
        upd = self.loc.tracker.update

        def timed(*a, **k):
            t0 = time.perf_counter()
            upd(*a, **k)
            self.tracker_ms.append((time.perf_counter() - t0) * 1e3)
        self.loc.tracker.update = timed
        shape = SHAPES[self.shape_i]
        self.bots = [W.Bot('enemy A', 'enemy', chaser, shape),
                     W.Bot('enemy B', 'enemy', goalie, shape, heading=lambda t: math.pi),
                     W.Bot('teammate', 'mate', mate, shape)]

    def us(self, t):
        """True pose at time t (interpolated history) — what the lidar simulator ray-casts from."""
        i = bisect.bisect_left(self.hist_t, t)
        if i <= 0:
            return self.hist_p[0]
        if i >= len(self.hist_t):
            return self.hist_p[-1]
        t0, t1 = self.hist_t[i - 1], self.hist_t[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        p0, p1 = self.hist_p[i - 1], self.hist_p[i]
        return tuple(p0[k] + a * (p1[k] - p0[k]) for k in range(3))

    # ------------------------------------------------------------ input
    def held(self, *names):
        return any(self.keys.get(n, 0) > 0 for n in names)

    def on_key(self, e):
        k = e.keysym.lower()
        first = self.keys.get(k, 0) <= 0
        self.keys[k] = self.keys.get(k, 0) + 1
        if not first:
            return                        # auto-repeat
        if k == 'escape':
            self.root.destroy()
        elif k == 'space':
            self.paused = not self.paused
        elif k == 'comma':
            self.speed = max(0.1, round(self.speed - 0.1, 1))
        elif k == 'period':
            self.speed = min(1.0, round(self.speed + 0.1, 1))
        elif k == 'r':
            self.robot_rel = not self.robot_rel
        elif k == 't':
            self.autospin = not self.autospin
        elif k == 'k':
            self.pose[0] = max(-70, min(70, self.pose[0] + random.choice((-1, 1)) * 30))
            self.pose[1] = max(-100, min(100, self.pose[1] + random.choice((-1, 1)) * 25))
        elif k == '1':
            for b in self.bots:
                b.fixed = None
        elif k == '2':
            for b in self.bots:
                b.fixed = b.pos(self.t)
        elif k == 'h':
            self.shape_i = (self.shape_i + 1) % len(SHAPES)
            for b in self.bots:
                b.shape = SHAPES[self.shape_i]
        elif k == 'g':
            self.posts = not self.posts
            ET.SCAN_ABOVE_GOALS = not self.posts
        elif k == 'p':
            self.show_pts = (self.show_pts + 1) % 3
        elif k == 'x':
            self.lidar.mixed = not self.lidar.mixed
        elif k == 'c':
            self.show_grid = not self.show_grid
        elif k == 'j':
            self.lidar.junk = not self.lidar.junk

    def on_key_up(self, e):
        k = e.keysym.lower()
        n = self.keys.get(k, 0)
        # X11 auto-repeat sends release+press pairs: only release if no press follows soon
        self.root.after(30, lambda: self.keys.__setitem__(k, 0) if self.keys.get(k, 0) == n else None)

    def on_press(self, e):
        x, y = to_field(e.x, e.y)
        best = min(self.bots, key=lambda b: math.hypot(b.pos(self.t)[0] - x, b.pos(self.t)[1] - y))
        if math.hypot(best.pos(self.t)[0] - x, best.pos(self.t)[1] - y) < 15:
            self.drag = best
            best.fixed = (x, y)

    def on_motion(self, e):
        if self.drag is not None:
            x, y = to_field(e.x, e.y)
            m = W.ROBOT_R
            self.drag.fixed = (max(-L.WALL_X + m, min(L.WALL_X - m, x)), max(-L.WALL_Y + m, min(L.WALL_Y - m, y)))

    def on_release(self, e):
        self.drag = None

    # ------------------------------------------------------------ simulation
    def step(self, dt):
        """Advance the true world by dt, feeding odometry at 200 Hz and lidar bytes as they come."""
        t_end = self.t + dt
        h = 1.0 / ODO_HZ
        while self.t + h <= t_end + 1e-9:
            self.t += h
            self.physics(h)
            x, y, th = self.pose
            self.hist_t.append(self.t)
            self.hist_p.append((x, y, th))
            vx, vy, om = self.vel
            if self.slipping:             # pushing a wall: wheels turn, robot doesn't move
                vx, vy = self.cmd[0], self.cmd[1]
            self.loc.on_odometry(self.t, *self.odo.sample(self.t, x, y, th, vx, vy, om, h))
        # teammate radio heartbeat: measured every 100 ms, arrives 50 ms later, 2 cm noise
        # teammate heartbeat (field_model.pack_heartbeat), every 100 ms, arrives 30 ms later.
        # The teammate isn't fully simulated: its pose and the enemies it reports are the
        # truth plus a few cm of noise, standing in for its own localiser and tracker.
        while self.next_radio <= self.t:
            tm, mate = self.next_radio, self.bots[2]
            mx, my = mate.pos(tm)
            mvx, mvy = mate.vel(tm)
            me = F.RobotState(mx + random.gauss(0, 1), my + random.gauss(0, 1), mate.theta(tm), mvx, mvy)
            seen = []
            for b in self.bots[:2]:
                ex, ey = b.pos(tm)
                if math.hypot(ex - mx, ey - my) < 200:
                    evx, evy = b.vel(tm)
                    seen.append(F.RobotState(ex + random.gauss(0, 3), ey + random.gauss(0, 3), None, evx, evy, 0.8))
            self.radio.append((tm + F.RADIO_DELAY, F.pack_heartbeat(2, True, me, seen)))
            self.next_radio += 0.1
        while self.radio and self.radio[0][0] <= self.t:
            self.wm.on_heartbeat(self.radio.popleft()[1], self.t)
        self.state = self.wm.update(self.t)      # the snapshot strategy code would get
        # lidar packets whose last point has happened
        while (self.lidar.k * 12 + 12) / 5000 <= self.t - 0.004:
            t_rx, raw, hits = self.lidar.next_packet(self.us, self.bots, self.posts)
            self.hits.extend(hits)
            self.loc.on_lidar_bytes(raw, t_rx)
        while self.hits and self.hits[0][0] < self.t - 0.11:
            self.hits.popleft()
        if len(self.hist_t) > 400:        # keep ~2 s of pose history
            del self.hist_t[:200], self.hist_p[:200]

    def physics(self, h):
        x, y, th = self.pose
        fast = self.held('shift_l', 'shift_r')
        sp = FAST if fast else DRIVE
        ix = (self.held('d') - self.held('a'))
        iy = (self.held('w') - self.held('s'))
        n = math.hypot(ix, iy) or 1.0
        cx, cy = sp * ix / n, sp * iy / n
        if self.robot_rel:                # robot frame: +y forward, +x right
            c, s = math.cos(th), math.sin(th)
            cx, cy = c * cx - s * cy, s * cx + c * cy
        om = 6.0 if self.autospin else TURN * (self.held('q') - self.held('e'))
        self.cmd = (cx, cy)
        vx, vy, _ = self.vel
        dvx, dvy = cx - vx, cy - vy
        dv = math.hypot(dvx, dvy)
        if dv > ACCEL * h:
            dvx, dvy = dvx * ACCEL * h / dv, dvy * ACCEL * h / dv
        vx, vy = vx + dvx, vy + dvy
        m = W.ROBOT_R
        nx, ny = x + vx * h, y + vy * h
        cx_, cy_ = max(-L.WALL_X + m, min(L.WALL_X - m, nx)), max(-L.WALL_Y + m, min(L.WALL_Y - m, ny))
        self.slipping = (cx_, cy_) != (nx, ny)
        if cx_ != nx: vx = 0.0
        if cy_ != ny: vy = 0.0
        self.pose = [cx_, cy_, L.wrap(th + om * h)]
        self.vel = [vx, vy, om]

    # ------------------------------------------------------------ drawing
    def _draw_static(self):
        cv = self.cv
        x0, y0 = to_px(-L.WALL_X, L.WALL_Y)
        x1, y1 = to_px(L.WALL_X, -L.WALL_Y)
        cv.create_rectangle(x0, y0, x1, y1, fill=FIELD, outline='')
        for gy, col in ((1, '#3b82f6'), (-1, '#facc15')):          # enemy goal top (blue), ours bottom
            a, b = to_px(-30, gy * L.WALL_Y)
            c, d = to_px(30, gy * (L.WALL_Y + 7.4))
            cv.create_rectangle(a, b, c, d, fill=col, outline='')
        cv.create_rectangle(x0, y0, x1, y1, outline=WALL, width=3)
        cx, cy = to_px(0, 0)
        cv.create_oval(cx - 30 * S, cy - 30 * S, cx + 30 * S, cy + 30 * S, outline='#2f5c43')
        cv.create_line(x0, cy, x1, cy, fill='#2f5c43')
        a, b = to_px(-L.WALL_X + 4, L.WALL_Y + PAD - 2)
        cv.create_text(a, b, anchor='nw', fill='#9aa7b4', font=('Segoe UI', 9), text='enemy goal (+y)')

    def belief(self):
        """Map a point the robot believes is at field (x, y) to where that is in reality:
        true_pose ∘ est^-1. Perfect localisation = identity."""
        T = L.compose(tuple(self.pose), L.inverse(self.loc.est))
        c, s = math.cos(T[2]), math.sin(T[2])
        return lambda x, y: (T[0] + c * x - s * y, T[1] + s * x + c * y)

    def draw_grid(self, B):
        """The field grid and WorldState.occupancy(), drawn where the robot believes they are."""
        cv = self.cv
        occ = self.state.occupancy()
        fill = {F.ME: '#5a6b7a', F.MATE: '#2f6b3f', F.ENEMY: '#7a2f36'}
        for row in range(F.ROWS):
            for col in range(F.COLS):
                code = int(occ[row, col])
                if code:
                    x0, y0 = -L.WALL_X + col * F.CELL_W, -L.WALL_Y + row * F.CELL_H
                    pts = []
                    for x, y in ((x0, y0), (x0 + F.CELL_W, y0), (x0 + F.CELL_W, y0 + F.CELL_H), (x0, y0 + F.CELL_H)):
                        pts += to_px(*B(x, y))
                    cv.create_polygon(*pts, fill=fill[code], outline='', tags='dyn')
        for col in range(F.COLS + 1):
            x = -L.WALL_X + col * F.CELL_W
            cv.create_line(*to_px(*B(x, -L.WALL_Y)), *to_px(*B(x, L.WALL_Y)), fill='#2a4a38', tags='dyn')
        for row in range(F.ROWS + 1):
            y = -L.WALL_Y + row * F.CELL_H
            cv.create_line(*to_px(*B(-L.WALL_X, y)), *to_px(*B(L.WALL_X, y)), fill='#2a4a38', tags='dyn')
        for col in range(0, F.COLS, 3):
            cv.create_text(*to_px(*B(F.cell_centre(col, 0)[0], -L.WALL_Y + 3)), text=str(col),
                           fill='#6f8f7d', font=('Segoe UI', 7), tags='dyn')
        for row in range(0, F.ROWS, 3):
            cv.create_text(*to_px(*B(-L.WALL_X + 3, F.cell_centre(0, row)[1])), text=str(row),
                           fill='#6f8f7d', font=('Segoe UI', 7), tags='dyn')

    def draw(self):
        cv = self.cv
        cv.delete('dyn')
        B = self.belief()
        if self.show_grid and not self.loc.lost:
            self.draw_grid(B)
        tracker = self.loc.tracker
        lost = self.loc.lost

        # goal side walls (only when the scan plane is below goal height)
        if self.posts:
            for px, ya, yb in W.POSTS:
                a, b = to_px(px, ya)
                c, d = to_px(px, yb)
                cv.create_line(a, b, c, d, fill='#f97316', width=3, tags='dyn')

        # believed field outline
        if not lost:
            pts = []
            for x, y in ((-L.WALL_X, -L.WALL_Y), (L.WALL_X, -L.WALL_Y), (L.WALL_X, L.WALL_Y), (-L.WALL_X, L.WALL_Y), (-L.WALL_X, -L.WALL_Y)):
                pts += to_px(*B(x, y))
            cv.create_line(*pts, fill=C_EST, dash=(6, 4), width=2, tags='dyn')
            for gy in (1, -1):
                a = to_px(*B(-30, gy * L.WALL_Y))
                b = to_px(*B(30, gy * L.WALL_Y))
                cv.create_line(*a, *b, fill=C_EST, width=4, tags='dyn')

        # lidar hits (true positions)
        if self.show_pts in (0, 2):
            for t, x, y, who in self.hits:
                X, Y = to_px(x, y)
                col = HIT_COL.get(who) or (C_HIT_MATE if self.bots[who].kind == 'mate' else C_HIT_ENEMY)
                r = 2.2 if who >= 0 else 1.3
                cv.create_rectangle(X - r, Y - r, X + r, Y + r, fill=col, outline='', tags='dyn')
        # what the tracker used (its own field frame, shown through the belief map)
        if self.show_pts in (1, 2) and tracker.last is not None and not lost:
            lt = tracker.last
            for x, y in zip(lt['fx'][lt['use']], lt['fy'][lt['use']]):
                X, Y = to_px(*B(x, y))
                cv.create_oval(X - 3, Y - 3, X + 3, Y + 3, outline='#ffb347', tags='dyn')
            for x, y in zip(lt['fx'][lt['mixed']], lt['fy'][lt['mixed']]):
                X, Y = to_px(*B(x, y))
                cv.create_line(X - 3, Y - 3, X + 3, Y + 3, fill='#d873ff', tags='dyn')
                cv.create_line(X - 3, Y + 3, X + 3, Y - 3, fill='#d873ff', tags='dyn')
            for c in tracker.clusters:
                X, Y = to_px(*B(c.x, c.y))
                cv.create_text(X, Y - 16, text=('A' if c.mode == 'arc' else 'F') + str(c.n),
                               fill='#ffb347', font=('Segoe UI', 8), tags='dyn')

        # true robots
        R = W.ROBOT_R * S
        for b in self.bots:
            x, y = b.pos(self.t)
            X, Y = to_px(x, y)
            col = C_ENEMY if b.kind == 'enemy' else C_MATE
            cv.create_oval(X - R, Y - R, X + R, Y + R, outline=col, width=2, fill='', tags='dyn')
            for cx, cy, r in b.circles(self.t):
                a, bb = to_px(cx, cy)
                cv.create_oval(a - r * S, bb - r * S, a + r * S, bb + r * S, fill=col, outline='', tags='dyn')
            th = b.theta(self.t)
            hx, hy = to_px(x - 11 * math.sin(th), y + 11 * math.cos(th))
            cv.create_line(X, Y, hx, hy, fill=col, tags='dyn')
            cv.create_text(X, Y + R + 8, text=b.name, fill=col, font=('Segoe UI', 8), tags='dyn')

        # tracked enemies, where the robot believes they are
        self.matches = {}
        if not lost:
            for e in self.loc.get_enemies(self.t):
                X, Y = to_px(*B(e.x, e.y))
                col = C_TRACK
                dash = (3, 3) if e.coasting else ()
                cv.create_oval(X - R, Y - R, X + R, Y + R, outline=col, width=2, dash=dash, tags='dyn')
                # 2-sigma ellipse of the position covariance
                w, V = np.linalg.eigh(e.cov[:2, :2])
                a, b2 = 2 * math.sqrt(max(w[1], 0)), 2 * math.sqrt(max(w[0], 0))
                ang = math.atan2(V[1, 1], V[0, 1])
                ell = []
                for k in range(25):
                    u = 2 * math.pi * k / 24
                    ex = e.x + a * math.cos(u) * math.cos(ang) - b2 * math.sin(u) * math.sin(ang)
                    ey = e.y + a * math.cos(u) * math.sin(ang) + b2 * math.sin(u) * math.cos(ang)
                    ell += to_px(*B(ex, ey))
                cv.create_line(*ell, fill=col, width=1, dash=(2, 4), tags='dyn')
                tx, ty = to_px(*B(e.x + 0.25 * e.vx, e.y + 0.25 * e.vy))
                cv.create_line(X, Y, tx, ty, fill=col, arrow='last', width=2, tags='dyn')
                cv.create_text(X + R + 4, Y - R, anchor='w', text=f'#{e.id}', fill=col,
                               font=('Segoe UI', 9, 'bold'), tags='dyn')
            for i, b in enumerate(self.bots):
                if b.kind != 'enemy':
                    continue
                x, y = b.pos(self.t)
                best = min(self.loc.get_enemies(self.t), key=lambda e: math.hypot(e.x - x, e.y - y), default=None)
                if best is not None and math.hypot(best.x - x, best.y - y) < 40:
                    err = math.hypot(best.x - x, best.y - y)
                    vx, vy = b.vel(self.t)
                    self.matches[b.name] = (best, err, math.hypot(best.vx - vx, best.vy - vy))
                    X0, Y0 = to_px(x, y)
                    X1, Y1 = to_px(*B(best.x, best.y))
                    cv.create_line(X0, Y0, X1, Y1, fill=C_ERR, tags='dyn')
                    cv.create_text((X0 + X1) / 2 + 6, (Y0 + Y1) / 2, anchor='w', text=f'{err:.1f}',
                                   fill=C_ERR, font=('Segoe UI', 8), tags='dyn')
            mt = tracker.teammate(self.t)
            if mt is not None:
                X, Y = to_px(*B(mt.x, mt.y))
                cv.create_line(X - 6, Y, X + 6, Y, fill=C_MATE, width=2, tags='dyn')
                cv.create_line(X, Y - 6, X, Y + 6, fill=C_MATE, width=2, tags='dyn')

        # enemies we only know from the teammate's heartbeat (ours can't see them right now)
        for e in self.state.enemies:
            if e.source == 'teammate':
                X, Y = to_px(*B(e.x, e.y))
                cv.create_oval(X - R, Y - R, X + R, Y + R, outline='#c4a7ff', width=2, dash=(1, 3), tags='dyn')
                cv.create_text(X + R + 4, Y - R, anchor='w', text='via mate', fill='#c4a7ff',
                               font=('Segoe UI', 8), tags='dyn')

        # our robot: true (solid), and the lidar beam direction
        x, y, th = self.pose
        X, Y = to_px(x, y)
        cv.create_oval(X - R, Y - R, X + R, Y + R, outline=C_US, width=2, tags='dyn')
        hx, hy = to_px(x - 13 * math.sin(th), y + 13 * math.cos(th))
        cv.create_line(X, Y, hx, hy, fill=C_US, width=3, tags='dyn')
        a = th - math.radians((3600 * self.t) % 360)
        bx, by = to_px(x - 40 * math.sin(a), y + 40 * math.cos(a))
        cv.create_line(X, Y, bx, by, fill='#5d6b78', tags='dyn')
        if lost:
            cv.create_text(CW / 2, 40, text='LOST — relocalising', fill='#ff6b6b',
                           font=('Segoe UI', 20, 'bold'), tags='dyn')

    def draw_hud(self):
        h = self.hud
        h.delete('all')
        y = 14

        def line(text, col='#dfe6ee', size=10, bold=False, gap=18):
            nonlocal y
            h.create_text(12, y, anchor='nw', text=text, fill=col, font=('Segoe UI', size, 'bold' if bold else 'normal'))
            y += gap

        def swatch(col, text, dashed=False):
            nonlocal y
            h.create_line(14, y + 7, 40, y + 7, fill=col, width=3, dash=(5, 3) if dashed else ())
            h.create_text(48, y, anchor='nw', text=text, fill='#c8d1da', font=('Segoe UI', 9))
            y += 16

        line(f't = {self.t:6.2f} s    speed {self.speed:.1f}x' + ('    PAUSED' if self.paused else ''), bold=True)
        est, (x, y_, th) = self.loc.est, self.pose
        if self.loc.lost:
            line('Localiser: LOST (relocalising)', '#ff6b6b', bold=True)
        else:
            line(f'Localiser: pos err {math.hypot(est[0] - x, est[1] - y_):4.1f} cm   heading err '
                 f'{math.degrees(abs(L.wrap(est[2] - th))):4.1f} deg')
            st = self.loc.stats
            line(f'   conf {self.loc.conf:.2f}   inliers {st.get("inlier", 0):.0%}   walls seen x{st.get("x_wall", 0)} y{st.get("y_wall", 0)}', '#9aa7b4', 9)
        if self.slipping:
            line('   pushing a wall: wheels slip, odometry lies', '#ffb347', 9)
        y += 4
        line('Enemy tracker', bold=True)
        ms = list(self.tracker_ms)
        if ms:
            line(f'   {np.mean(ms):.2f} ms / batch (max {max(ms):.2f})', '#9aa7b4', 9)
        cl = self.loc.tracker.clusters
        line(f'   last batch: {sum(c.mode == "arc" for c in cl)} body arcs, {sum(c.mode == "fragment" for c in cl)} fragments'
             f'{", teammate seen" if any(c.teammate for c in cl) else ""}', '#9aa7b4', 9)
        for b in self.bots:
            if b.kind != 'enemy':
                continue
            m = self.matches.get(b.name)
            if m is None:
                line(f'   {b.name}: not tracked', '#ff6b6b')
            else:
                e, err, verr = m
                line(f'   {b.name}: #{e.id}  err {err:4.1f} cm  v err {verr:4.0f} cm/s{"  (coasting)" if e.coasting else ""}')
        y += 4
        line('World state  (what strategy code gets)', bold=True)
        st = self.state

        def rs(label, r):
            if r is None:
                line(f'   {label}: —', '#9aa7b4', 9, gap=15)
                return
            hd = '   —  ' if r.heading is None else f'{r.heading_deg:+6.1f}°'
            line(f'   {label:<9} ({r.x:+6.1f}, {r.y:+6.1f}) cm  cell {r.cell}  hdg {hd}  '
                 f'{r.speed:3.0f} cm/s', '#dfe6ee' if st.valid else '#ff6b6b', 9, gap=15)
        rs('me', st.me)
        rs('mate', st.mate)
        for i in range(ET.MAX_ENEMIES):
            e = st.enemies[i] if i < len(st.enemies) else None
            rs(f'enemy {i + 1}' + ('' if e is None or e.source == 'lidar' else '*'), e)
        if any(e.source == 'teammate' for e in st.enemies):
            line("   * known only from the teammate's heartbeat", '#c4a7ff', 8, gap=14)
        line(f'   grid {F.COLS}x{F.ROWS} cells of {F.CELL_W:.1f} cm   [C]', '#9aa7b4', 9)
        y += 4
        line('Scene', bold=True)
        line(f'   robots at scan height: {SHAPES[self.shape_i]}      [H]', '#9aa7b4', 9)
        line(f'   scan plane {"BELOW goal height (posts visible)" if self.posts else "above goal height"}   [G]', '#9aa7b4', 9)
        line(f'   mixed pixels {"on" if self.lidar.mixed else "off"} [X]   junk {"on" if self.lidar.junk else "off"} [J]', '#9aa7b4', 9)
        line(f'   points: {("true hits", "tracker view", "both")[self.show_pts]}   [P]', '#9aa7b4', 9)
        line(f'   driving: {"robot-relative" if self.robot_rel else "field-relative"} [R]   spin {"on" if self.autospin else "off"} [T]', '#9aa7b4', 9)
        y += 6
        line('Legend', bold=True)
        swatch(C_US, 'our robot (true)')
        swatch(C_EST, 'field where the robot thinks it is', True)
        swatch(C_ENEMY, 'enemy, true position + handles / bits')
        swatch(C_HIT_ENEMY, 'lidar hit: on an enemy')
        swatch(C_TRACK, 'enemy track (dashed while coasting)', True)
        swatch(C_ERR, 'track error the WorldModel gets')
        swatch(C_MATE, 'teammate (+ = its tracked position)')
        swatch(C_HIT_MATE, 'lidar hit: on the teammate')
        swatch(HIT_COL[W.HIT_WALL], 'lidar hit: wall')
        swatch(HIT_COL[W.HIT_MIXED], 'lidar hit: mixed pixel (x = tracker dropped it)')
        swatch(HIT_COL[W.HIT_JUNK], 'lidar hit: junk')
        swatch('#ffb347', 'tracker view: obstacle points, A=arc F=fragment')
        y += 6
        line('Keys', bold=True)
        for k in ('WASD drive, Shift fast, Q/E turn', 'K kidnap   1/2 robots move/stand   drag = place',
                  ', . speed   Space pause   Esc quit'):
            line('   ' + k, '#9aa7b4', 9, gap=15)

    # ------------------------------------------------------------ loop
    def frame(self):
        now = time.perf_counter()
        real = min(now - self.last_wall, 0.05)
        self.last_wall = now
        if not self.paused:
            self.step(real * self.speed)
        self.draw()
        self.draw_hud()
        self.root.after(max(1, int(33 - (time.perf_counter() - now) * 1000)), self.frame)


def make_root():
    """Tk root, DPI-aware on Windows, with the field scaled to fit the screen height."""
    global S, CW, CH
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # crisp, true-pixel drawing
    except Exception:
        pass
    root = tk.Tk()
    S = max(1.5, min(4.0, (root.winfo_screenheight() - 120) / (2 * FH)))
    CW, CH = int(2 * FW * S), int(2 * FH * S)
    return root


def main():
    root = make_root()
    Viewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
