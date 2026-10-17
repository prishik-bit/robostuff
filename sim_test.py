"""Synthetic D500 stream: robot doing a figure-8 at up to ~3 m/s while spinning 6 rad/s,
an opponent robot occluding walls, 2% junk returns, goal recesses, packet jitter,
5% odometry scale error, and a referee 'pick-up and move' at t=4 s.

Part 2 (enemy tracking): 2 enemies (a chaser running at our goal, a goalie shuffling in
front of theirs) + 1 teammate on the radio, occluding each other, ray-cast with their
scan-height shape from sim_world.py (just the two handles, handles + small bits, or a full
body), with mixed pixels at depth edges; optional goal side walls for a scan plane below
goal height; and a scripted full-occlusion test."""
import math, random, struct, sys, time
import numpy as np
import lidar_loc as L
import enemy_tracker as ET
import sim_world as W_
import field_model as F

random.seed(1); np.random.seed(1)
W = 1.68
def truth(t, kid):
    x = 55 * math.sin(W * t); y = 85 * math.sin(2 * W * t)
    if kid and t > 4.0: x += 25; y -= 30           # kidnapped
    return x, y, 6.0 * t
def vel(t):
    return 55 * W * math.cos(W * t), 170 * W * math.cos(2 * W * t)

def ray(x, y, dx, dy, opp):
    tx = ((L.WALL_X if dx > 0 else -L.WALL_X) - x) / dx if abs(dx) > 1e-9 else 1e9
    ty = ((L.WALL_Y if dy > 0 else -L.WALL_Y) - y) / dy if abs(dy) > 1e-9 else 1e9
    d = min(tx, ty)
    if ty < tx and abs(x + dx * ty) < 30: d += 7.4 / abs(dy)   # into goal box
    ox, oy = opp[0] - x, opp[1] - y                  # opponent: 9 cm circle
    b = ox * dx + oy * dy; c = ox * ox + oy * oy - 81
    disc = b * b - c
    if disc > 0 and b - math.sqrt(disc) > 0: d = min(d, b - math.sqrt(disc))
    return d

def run(deskew_on, kid):
    L.deskew_impl = L.deskew
    loc = L.Localiser(pose=(15, -15, 0.0))           # deliberately wrong start
    if not deskew_on:
        L.deskew = lambda odom, t, qx, qy, t_ref: (qx, qy)
    events = []
    for k in range(int(6.0 * 200)):                  # odometry 200 Hz
        events.append((k / 200, 'odo'))
    pkts = []
    n = 0
    for k in range(int(6.0 * 5000) // 12):           # lidar 5 kHz, 12 pts/packet
        ts = [(k * 12 + i) / 5000 for i in range(12)]
        angs, dists = [], []
        for t in ts:
            x, y, th = truth(t, kid)
            ang = (3600 * t) % 360
            a = th - math.radians(ang)               # CW lidar -> CCW robot -> field
            d = ray(x, y, -math.sin(a), math.cos(a), (-30 + 20 * math.sin(t), 40))
            d += random.gauss(0, 1.0)
            if random.random() < 0.02: d = random.uniform(5, 300)   # junk
            angs.append(ang); dists.append(int(max(d, 0) * 10))
        body = [0x54, 0x2C, 3600, int(angs[0] * 100)]
        for d in dists: body += [min(d, 65535), 200]
        body += [int(angs[-1] * 100), int(round(ts[-1] * 1000)) % 30000]
        raw = struct.pack('<BBHH' + 'HB' * 12 + 'HH', *body)
        raw += bytes([L.crc8(raw)])
        if random.random() < 0.01: raw = raw[:20] + b'\x00' + raw[21:]   # corrupt
        events.append((ts[-1] + L.PKT_TX_S + random.uniform(0, 0.003), 'lid', raw))
    events.sort(key=lambda e: e[0])
    yaw_bias = 0.0
    errs = []
    for e in events:
        t = e[0]
        if e[1] == 'odo':
            x, y, th = truth(t, False)               # odometry never sees the kidnap
            vx, vy = vel(t)
            c, s = math.cos(th), math.sin(th)
            vbx = (c * vx + s * vy) * 1.05 + random.gauss(0, 3)
            vby = (-s * vx + c * vy) * 1.05 + random.gauss(0, 3)
            yaw_bias += 0.01 * 6.0 / 200
            loc.on_odometry(t, vbx, vby, th + yaw_bias)
        else:
            loc.on_lidar_bytes(e[2], t)
            if t > 0.5:
                x, y, th = truth(t, kid)
                ex = math.hypot(loc.est[0] - x, loc.est[1] - y)
                eth = abs(L.wrap(loc.est[2] - th))
                errs.append((t, ex, eth, loc.lost))
    L.deskew = L.deskew_impl
    a = np.array(errs)
    good = a[(a[:, 0] < 3.9) | (a[:, 0] > 4.6)]
    print(f"deskew={deskew_on!s:5} kidnap={kid!s:5}  pos err mean {good[:,1].mean():5.1f} cm"
          f"  p95 {np.percentile(good[:,1],95):5.1f} cm   heading err mean "
          f"{math.degrees(good[:,2].mean()):4.1f} deg   crc_bad={loc.parser.bad_crc}")
    if kid:
        post = a[a[:, 0] > 4.0]
        rec = post[(post[:, 1] < 5)]
        print(f"   kidnap at 4.00 s -> back within 5 cm at {rec[0,0]:.2f} s")



# ============================ Part 2: enemy tracking ============================
# Robots are ray-cast with their scan-height shape (sim_world.SHAPES): mostly just the two
# handles, sometimes handles + small bits, and full bodies for comparison.
R_BOT = ET.ROBOT_R


def simulate(T, us, bots, posts=False, seed=2, radio=True, world=False):
    """us(t) -> true (x, y, th). bots: [sim_world.Bot], kind 'enemy' / 'mate'.
    world=True: run a field_model.WorldModel on top, with the teammate sending real 32-byte
    heartbeats (its pose and the enemies it sees: truth + noise, standing in for its own
    localiser and tracker), and record the WorldState strategy code would get.
    Returns per-update records, per-bot ray-hit times, tracker timings, the localiser."""
    random.seed(seed); np.random.seed(seed)
    lidar, odo = W_.LidarSim(seed), W_.Odometry(seed + 1)
    x0, y0, th0 = us(0)
    loc = L.Localiser(pose=(x0 + 15, y0 - 15, th0))  # deliberately wrong start
    calls = []                                         # tracker inputs, replayed for timing
    upd, mset = loc.tracker.update, loc.tracker.set_teammate
    def logged(*a, **k):
        calls.append(('update', a, k)); return upd(*a, **k)
    def logged_mate(*a):
        calls.append(('mate', a, {})); return mset(*a)
    loc.tracker.update, loc.tracker.set_teammate = logged, logged_mate
    events = [(k / 200, 'odo') for k in range(int(T * 200))]
    rr = random.Random(seed + 2)
    wm = F.WorldModel(loc, robot_id=1) if world else None
    if radio:
        mate_bot = [b for b in bots if b.kind == 'mate'][0]
        for k in range(int(T * 10)):                   # 10 Hz heartbeat, arrives 50 ms late, 2 cm noise
            tm = k / 10
            mx, my = mate_bot.pos(tm)
            if not world:
                events.append((tm + 0.05, 'radio', (tm, mx + rr.gauss(0, 2), my + rr.gauss(0, 2))))
                continue
            mvx, mvy = mate_bot.vel(tm)
            me = F.RobotState(mx + rr.gauss(0, 1), my + rr.gauss(0, 1), mate_bot.theta(tm), mvx, mvy)
            seen_by_mate = [F.RobotState(b.pos(tm)[0] + rr.gauss(0, 3), b.pos(tm)[1] + rr.gauss(0, 3), None,
                                         *b.vel(tm), 0.8)
                            for b in bots if b.kind == 'enemy' and math.hypot(b.pos(tm)[0] - mx, b.pos(tm)[1] - my) < 200]
            events.append((tm + F.RADIO_DELAY, 'heartbeat', F.pack_heartbeat(2, True, me, seen_by_mate)))
    seen = [[] for _ in bots]
    for _ in range(int(T * 5000) // 12):
        t_rx, raw, hits = lidar.next_packet(us, bots, posts)
        for t, hx, hy, who in hits:
            if who >= 0: seen[who].append(t)
        events.append((t_rx, 'lid', raw))
    events.sort(key=lambda e: e[0])
    rec = []
    for e in events:
        t = e[0]
        if e[1] == 'odo':
            x, y, th = us(t)
            h = 1e-4
            vx, vy = (us(t + h)[0] - us(t - h)[0]) / (2 * h), (us(t + h)[1] - us(t - h)[1]) / (2 * h)
            om = (us(t + h)[2] - us(t - h)[2]) / (2 * h)
            loc.on_odometry(t, *odo.sample(t, x, y, th, vx, vy, om, 1 / 200))
        elif e[1] == 'radio':
            loc.set_teammate(*e[2])
        elif e[1] == 'heartbeat':
            wm.on_heartbeat(e[2], t)
        else:
            n_before = len(calls)
            loc.on_lidar_bytes(e[2], t)
            if not any(c[0] == 'update' for c in calls[n_before:]) and not loc.lost:
                continue                               # sample once per tracker update
            x, y, th = us(t)
            rec.append({'t': t, 'lost': loc.lost, 'pos_err': math.hypot(loc.est[0] - x, loc.est[1] - y),
                        'truth': [(b.pos(t), b.vel(t)) for b in bots],
                        'enemies': loc.get_enemies(t),
                        'modes': [c.mode for c in loc.tracker.clusters if not c.teammate],
                        'tracks': [(tr.id, tr.x[0], tr.x[1]) for tr in loc.tracker.tracks if tr.confirmed]})
            if wm is not None:
                rec[-1]['state'] = wm.update(t)
                rec[-1]['us'] = (x, y, th)
                rec[-1]['theta'] = [b.theta(t) for b in bots]
    return rec, seen, replay_times(calls), loc


REPLAYS = 5


def replay_times(calls):
    """Time EnemyTracker.update on the recorded inputs, away from the simulator's own work."""
    runs = []
    for _ in range(REPLAYS):
        tr = ET.EnemyTracker()
        out = []
        for kind, a, k in calls:
            if kind == 'mate':
                tr.set_teammate(*a)
                continue
            t0 = time.perf_counter()
            tr.update(*a, **k)
            out.append(time.perf_counter() - t0)
        runs.append(out)
    return np.median(np.array(runs), axis=0)          # per batch: median over replays (drops OS jitter)


def evaluate(name, rec, seen, times, bots, created, t_eval=1.0, match_r=25.0, brief=False):
    enemy_idx = [i for i, b in enumerate(bots) if b.kind == 'enemy']
    mate_idx = [i for i, b in enumerate(bots) if b.kind == 'mate']
    t_found = next(r['t'] for r in rec if not r['lost'])
    ev = [r for r in rec if r['t'] > t_eval and not r['lost']]
    print(f"\n[{name}]")
    if brief:
        report_false(ev, enemy_idx, mate_idx, created)
        return
    modes = [m for r in ev for m in r['modes']]
    print(f"  localiser found itself at {t_found:.2f} s; {len(ev)} tracker updates evaluated after {t_eval:.1f} s;"
          f" robot measurements: {modes.count('arc')} body arcs, {modes.count('fragment')} handle/bit fragments")
    pe_loc = [r['pos_err'] for r in ev]
    print(f"  self-localisation   pos err mean {np.mean(pe_loc):.2f} cm   p95 {np.percentile(pe_loc, 95):.2f} cm")
    print(f"  tracker time/batch  mean {times.mean()*1e3:.3f} ms   p99 {np.percentile(times, 99)*1e3:.3f} ms   "
          f"max {times.max()*1e3:.3f} ms   (median of {REPLAYS} replays per batch, this machine)")

    # Per enemy: the nearest output enemy within match_r of the truth is "its" track.
    match = {i: [] for i in enemy_idx}
    for r in ev:
        for i in enemy_idx:
            (tx, ty), (tvx, tvy) = r['truth'][i]
            best = min(r['enemies'], key=lambda e: math.hypot(e.x - tx, e.y - ty), default=None)
            if best is not None and math.hypot(best.x - tx, best.y - ty) < match_r:
                match[i].append((r, best.id, math.hypot(best.x - tx, best.y - ty),
                                 math.hypot(best.vx - tvx, best.vy - tvy)))
            else:
                match[i].append((r, None, None, None))
    owner = {}
    for i in enemy_idx:
        for r, tid, _, _ in match[i]:
            if tid is not None:
                owner.setdefault(tid, set()).add(i)

    def in_contact(r, i):                    # another robot within touching distance (+ margin)
        (x, y), _ = r['truth'][i]
        return any(math.hypot(x - r['truth'][j][0][0], y - r['truth'][j][0][1]) < 2 * R_BOT + 12
                   for j in range(len(bots)) if j != i)

    for i in enemy_idx:
        m = [v for v in match[i] if v[1] is not None]
        if not m:
            print(f"  {bots[i].name:<15} never tracked")
            continue
        pe = np.array([v[2] for v in m]); ve = np.array([v[3] for v in m])
        first = next((r['t'] for r in rec if not r['lost'] and any(
            math.hypot(e.x - r['truth'][i][0][0], e.y - r['truth'][i][0][1]) < match_r
            for e in r['enemies'])), math.nan)
        changes = swaps = swaps_contact = 0
        prev = None
        for r, tid, _, _ in m:
            if prev is not None and tid != prev:
                changes += 1
                if len(owner.get(tid, ())) > 1 or len(owner.get(prev, ())) > 1:
                    swaps += 1                       # an id that also belonged to the other enemy
                    swaps_contact += in_contact(r, i)
            prev = tid
        print(f"  {bots[i].name:<15} pos err mean {pe.mean():.2f} cm  p95 {np.percentile(pe, 95):.2f} cm   "
              f"vel err mean {ve.mean():.1f} cm/s  p95 {np.percentile(ve, 95):.1f} cm/s   "
              f"tracked {100 * len(m) / len(match[i]):.0f}% of updates")
        print(f"  {'':<15} first confirmed {first - t_found:.2f} s after localisation   "
              f"ID changes {changes}, swaps {swaps} (during contact {swaps_contact})")
        # Full occlusions: stretches with no ray hitting this robot for over one revolution.
        h = np.array(sorted(seen[i]))
        h = h[h > t_eval]
        gaps = [(a, b) for a, b in zip(h[:-1], h[1:]) if b - a > 0.15]
        kept, lost = [], []
        for a, b in gaps:
            before = [tid for r, tid, _, _ in match[i] if a - 0.1 < r['t'] <= a and tid is not None]
            after = [tid for r, tid, _, _ in match[i] if b < r['t'] <= b + 0.25 and tid is not None]
            if before and after and before[-1] == after[0]:
                kept.append(b - a)
            elif before:
                lost.append(b - a)
        print(f"  {'':<15} full occlusions > 0.15 s: {len(gaps)}; same track kept through {len(kept)} "
              f"(longest {max(kept, default=0):.2f} s); re-acquired as a new id after {len(lost)}"
              f"{' (' + ', '.join(f'{x:.2f} s' for x in lost) + ')' if lost else ''}")
    report_false(ev, enemy_idx, mate_idx, created)


def report_false(ev, enemy_idx, mate_idx, created):
    """False tracks: confirmed (internal) tracks whose median distance to the nearest enemy > 12 cm."""
    hist = {}
    for r in ev:
        for tid, x, y in r['tracks']:
            de = min(math.hypot(x - r['truth'][i][0][0], y - r['truth'][i][0][1]) for i in enemy_idx)
            dm = min((math.hypot(x - r['truth'][i][0][0], y - r['truth'][i][0][1]) for i in mate_idx),
                     default=1e9)
            dg = math.hypot(abs(x) - 30.0, abs(y) - 113.0)      # distance to a goal side wall's middle
            hist.setdefault(tid, []).append((de, dm, dg))
    false = {'goal structure': 0, 'teammate': 0, 'other': 0}
    for tid, v in hist.items():
        v = np.array(v)
        if np.median(v[:, 0]) > 12:                    # real tracks sit a few cm from their enemy
            kind = ('teammate' if np.median(v[:, 1]) < 25 else
                    'goal structure' if np.median(v[:, 2]) < 20 else 'other')
            false[kind] += 1
    print(f"  false confirmed tracks: {sum(false.values())} "
          f"({', '.join(f'{k} {n}' for k, n in false.items())});  tracks ever born: {created}")


def evaluate_world(name, rec, bots, t_eval=1.0, match_r=25.0):
    """Check the WorldState snapshots (field_model) against the truth."""
    ev = [r for r in rec if r['t'] > t_eval and r['state'].valid]
    deg = lambda a: abs(math.degrees(L.wrap(a)))
    print(f"\n[{name}]  {len(ev)} snapshots")
    me_p = [math.hypot(r['state'].me.x - r['us'][0], r['state'].me.y - r['us'][1]) for r in ev]
    me_h = [deg(r['state'].me.heading - r['us'][2]) for r in ev]
    print(f"  me        pos err mean {np.mean(me_p):.2f} cm  p95 {np.percentile(me_p, 95):.2f}   "
          f"heading err mean {np.mean(me_h):.2f} deg  p95 {np.percentile(me_h, 95):.2f}   cell right {100 * np.mean([F.cell_of(*r['us'][:2]) == r['state'].me.cell for r in ev]):.0f}%")
    mi = [i for i, b in enumerate(bots) if b.kind == 'mate'][0]
    mm = [(r['state'].mate, r['truth'][mi][0], r['theta'][mi]) for r in ev if r['state'].mate is not None]
    mp = [math.hypot(m.x - p[0], m.y - p[1]) for m, p, _ in mm]
    mh = [deg(m.heading - th) for m, _, th in mm]
    print(f"  teammate  present {100 * len(mm) / len(ev):.0f}%   pos err mean {np.mean(mp):.2f} cm  p95 {np.percentile(mp, 95):.2f}   "
          f"heading err mean {np.mean(mh):.2f} deg  (radio: 10 Hz, quantised to 1 mm)")
    for i, b in enumerate(bots):
        if b.kind != 'enemy':
            continue
        pe, he, has_h, moving, src, occ = [], [], 0, 0, {'lidar': 0, 'teammate': 0}, []
        for r in ev:
            (tx, ty), (tvx, tvy) = r['truth'][i]
            e = min(r['state'].enemies, key=lambda e: math.hypot(e.x - tx, e.y - ty), default=None)
            if e is None or math.hypot(e.x - tx, e.y - ty) > match_r:
                continue
            pe.append(math.hypot(e.x - tx, e.y - ty))
            src[e.source] += 1
            col, row = F.cell_of(tx, ty)
            occ.append(r['state'].occupancy()[row, col] == F.ENEMY)
            if math.hypot(tvx, tvy) >= F.HEADING_MIN_SPEED:      # truth has a direction of travel
                moving += 1
                if e.heading is not None:
                    has_h += 1
                    he.append(deg(e.heading - F.heading_of(tvx, tvy)))
        print(f"  {b.name:<15} in state {100 * len(pe) / len(ev):.0f}% ({src['teammate']} snapshots only via teammate)   "
              f"pos err mean {np.mean(pe):.2f} cm  p95 {np.percentile(pe, 95):.2f}   own cell marked {100 * np.mean(occ):.0f}%")
        if moving:
            print(f"  {'':<15} heading (= direction of travel) given {100 * has_h / moving:.0f}% of the time it moves; "
                  f"err mean {np.mean(he):.1f} deg  p95 {np.percentile(he, 95):.1f}")


# --- scene: all paths in field cm, our goal at -y ---
def us_path(t):          # our striker: lower half, fast, spinning 6 rad/s
    return 55 * math.sin(1.4 * t), -40 + 40 * math.sin(2.8 * t), 6.0 * t
def chaser(t):           # enemy striker running at our goal and back, with hard wiggles
    return 45 * math.sin(2.1 * t + 0.5) + 8 * math.sin(7 * t), 5 - 65 * math.sin(1.6 * t)
def goalie(t):           # enemy goalie shuffling in front of their goal
    return 24 * math.sin(2.5 * t) + 4 * math.sin(7.3 * t), 97 + 3 * math.sin(4 * t)
def mate(t):             # our teammate wandering midfield, crossing sight lines
    return -45 + 30 * math.sin(0.9 * t), 35 + 45 * math.sin(1.3 * t + 1)


def scene(enemy_shape, mate_shape='handles'):
    # The goalie faces our half (theta = pi), so its handles sit side to side across the goal.
    return [W_.Bot('enemy A chaser', 'enemy', chaser, enemy_shape),
            W_.Bot('enemy B goalie', 'enemy', goalie, enemy_shape, heading=lambda t: math.pi),
            W_.Bot('teammate', 'mate', mate, mate_shape)]


def main():
    run(True, False)
    run(False, False)
    run(True, True)

    print("\n================ enemy tracking ================")
    ET.SCAN_ABOVE_GOALS = True
    for shape, label in (('handles', 'only the two handles reach the scan plane'),
                         ('bits', 'handles + small bits'),
                         ('body', 'full 18 cm bodies (for comparison)')):
        bots = scene(shape)
        rec, seen, times, loc = simulate(10.0, us_path, bots)
        evaluate(f'scan above goal height, {label}', rec, seen, times, bots, loc.tracker.created)

    ET.SCAN_ABOVE_GOALS = False
    bots = scene('handles')
    rec, seen, times, loc = simulate(10.0, us_path, bots, posts=True)
    evaluate('scan below goal height: goal side walls visible, goal-structure filter on (handles)',
             rec, seen, times, bots, loc.tracker.created)
    ET.SCAN_ABOVE_GOALS = True                           # control: same scene, filter off
    bots = scene('handles')
    rec, seen, times, loc = simulate(10.0, us_path, bots, posts=True)
    evaluate('control: same goal side walls, goal-structure filter OFF (compare tracks ever born)',
             rec, seen, times, bots, loc.tracker.created, brief=True)

    # --- the WorldState strategy code gets (field_model.py), with real radio heartbeats ---
    print("\n================ world model (field_model.py) ================")
    print(f"  grid {F.COLS} x {F.ROWS} cells of {F.CELL_W:.2f} x {F.CELL_H:.2f} cm; heartbeat {F.HB_SIZE} bytes")
    bots = scene('handles')
    rec, seen, times, loc = simulate(10.0, us_path, bots, world=True)
    evaluate_world('handles robots, teammate heartbeats at 10 Hz', rec, bots)

    # --- scripted full occlusion: teammate parks between us and a near-stationary enemy ---
    print("\n[full-occlusion test, handles enemy] we sit at (0, -60) spinning 3 rad/s, enemy drifts"
          "\n  around (10, 50), teammate (a full body, so it really blocks) parks on the line of sight")
    for hold in (0.3, 0.6, 1.0, 1.4, 2.0):
        t_in, t_out = 2.0, 2.0 + hold
        def blocker(t, t_in=t_in, t_out=t_out):
            if t < t_in - 0.4: return -80.0, -5.0
            if t < t_in: return -80 + (t - (t_in - 0.4)) / 0.4 * 84.5, -5.0
            if t < t_out: return 4.5, -5.0
            return 4.5 + (t - t_out) / 0.4 * 84.5, -5.0
        bots = [W_.Bot('enemy', 'enemy', lambda t: (10.0 + 4 * math.sin(t), 50.0), 'handles',
                       heading=lambda t: 0.4 * t),
                W_.Bot('teammate', 'mate', blocker, 'body')]
        rec, seen, times, loc = simulate(t_out + 1.2, lambda t: (0.0, -60.0, 3.0 * t), bots, seed=3)
        def ids(a, b):
            return {e.id for r in rec if a < r['t'] <= b for e in r['enemies']
                    if math.hypot(e.x - 10 - 4 * math.sin(r['t']), e.y - 50) < 25}
        h = np.array(seen[0])
        h = h[h > t_in - 0.5]
        blind = max((b - a for a, b in zip(h[:-1], h[1:])), default=0.0)
        before, after = ids(t_in - 0.5, t_in - 0.2), ids(t_out + 0.6, t_out + 1.2)
        kept = bool(before) and before == after
        print(f"  parked {hold:.1f} s -> no ray on the enemy for {blind:.2f} s: "
              f"{'same track kept' if kept else 'track re-born with a new id'}  ids {sorted(before)} -> {sorted(after)}")


if __name__ == '__main__':
    main()
