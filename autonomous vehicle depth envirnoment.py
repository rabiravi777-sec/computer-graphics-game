"""
Autonomous Vehicle Environment Depth Visualizer
Computer Graphics concept: Z-Buffer

Features
  * Infinite road built from 25-unit chunks (created ahead, deleted behind)
  * Moving traffic that follows lanes and keeps a safe distance
  * Autopilot: cruises, brakes for the car in front, changes lane when it is free
  * Depth map view (near = white, far = black), the Z-buffer idea made visible

Controls
    Up / Down arrows : drive forward / backward  (hold SHIFT to go faster)
    Left / Right     : steer
    A                : toggle AUTOPILOT on / off
    D                : toggle Normal view <-> Depth map
    Esc              : quit
"""
from ursina import *
import math
import random

app = Ursina()

window.title = "Autonomous Vehicle Environment Depth Visualizer"
window.size = (1200, 800)
window.borderless = False
window.fps_counter.enabled = False


# ---------------------------------------------------------------- helpers
def rgb255(r, g, b):
    """Colour helper that works on both old and new Ursina versions."""
    return color.rgb32(r, g, b) if hasattr(color, "rgb32") else color.rgb(r, g, b)


SKY = rgb255(135, 190, 225)
window.color = SKY

GROUND_COLOR = rgb255(50, 150, 60)
ROAD_COLOR = rgb255(65, 65, 70)
BUILDING_COLORS = [rgb255(190, 170, 140), rgb255(130, 160, 200),
                   rgb255(200, 120, 110), rgb255(160, 160, 170)]
CAR_COLORS = [(color.red, "Red car"), (color.blue, "Blue car"),
              (color.orange, "Orange car"), (color.white, "White car"),
              (color.gray, "Gray car")]


def box(store, **kwargs):
    """Create an unlit cube, remember its normal colour, register it in `store`."""
    e = Entity(model="cube", unlit=True, **kwargs)
    e.base_color = e.color
    store.append(e)
    return e


def create_car(store, x, z, body_color, parent=None):
    kw = {"parent": parent} if parent is not None else {}
    car = Entity(position=(x, 0, z), **kw)
    box(store, parent=car, position=(0, 0.5, 0), scale=(1.7, 0.7, 2.5), color=body_color)
    box(store, parent=car, position=(0, 1.05, -0.1), scale=(1.1, 0.5, 1.2), color=color.azure)
    for wx in (-0.9, 0.9):
        for wz in (-0.75, 0.75):
            box(store, parent=car, position=(wx, 0.25, wz),
                scale=(0.25, 0.4, 0.5), color=color.black)
    return car


# ------------------------------------------------------- infinite world
CHUNK = 25          # length of one world chunk (z direction)
AHEAD = 200         # how far ahead of the car the world is generated
BEHIND = 45         # how far behind the car the world is kept
LANE = 3.5          # lane centres are x = -LANE and x = +LANE
chunks = {}         # chunk index -> {"root", "boxes", "obstacles"}


def build_chunk(i):
    """Create chunk number i, covering z = i*CHUNK ... (i+1)*CHUNK."""
    rng = random.Random(i * 7919 + 13)          # same chunk -> same content
    z0 = i * CHUNK
    root = Entity(position=(0, 0, z0))
    boxes, obstacles = [], []

    # Ground and road, cut into 5-unit pieces so the depth gradient is smooth
    for k in range(5):
        zc = k * 5 + 2.5
        box(boxes, parent=root, position=(0, -0.6, zc), scale=(200, 1, 5), color=GROUND_COLOR)
        box(boxes, parent=root, position=(0, 0.05, zc), scale=(14, 0.2, 5), color=ROAD_COLOR)
        box(boxes, parent=root, position=(0, 0.16, zc), scale=(0.15, 0.02, 2.5), color=color.white)

    # Buildings on both sides of the road
    for lz in (4, 16):
        for side in (-1, 1):
            h = rng.uniform(4, 10)
            box(boxes, parent=root,
                position=(side * rng.uniform(9, 12), h / 2, lz + rng.uniform(-2, 2)),
                scale=(4, h, 4), color=rng.choice(BUILDING_COLORS))

    # Yellow obstacles on the road shoulders (first chunks stay clear)
    if i >= 0 and rng.random() < 0.5:
        lx = rng.choice((-1, 1)) * rng.uniform(5.3, 6.2)
        lz = rng.uniform(5, 20)
        box(boxes, parent=root, position=(lx, 0.7, lz), scale=(1.2, 1.2, 1.2), color=color.yellow)
        # object format: (x, z, half_x, half_z, label, speed)
        obstacles.append((lx, z0 + lz, 0.6, 0.6, "Yellow obstacle", 0))

    return {"root": root, "boxes": boxes, "obstacles": obstacles}


def manage_chunks():
    """Generate chunks in front of the ego car, delete the ones far behind."""
    first = math.floor((ego.z - BEHIND) / CHUNK)
    last = math.floor((ego.z + AHEAD) / CHUNK)
    for i in range(first, last + 1):
        if i not in chunks:
            chunks[i] = build_chunk(i)
    for i in list(chunks):
        if i < first or i > last:
            destroy(chunks.pop(i)["root"])


# ---------------------------------------------------------------- ego car
EGO_HX, EGO_HZ = 0.95, 1.25
ego_boxes = []
ego = create_car(ego_boxes, LANE, -10, color.magenta)
ego_speed = 0.0             # metres per second along +z
autopilot_on = False
target_x = LANE             # lane the autopilot is steering towards
status = "Manual"

# ----------------------------------------------------------------- camera
CAM_OFFSET = Vec3(0, 14, -22)
camera.fov = 70
camera.rotation = (28, 0, 0)
camera.clip_plane_far = 1000
camera.position = ego.position + CAM_OFFSET

# ---------------------------------------------------------------- traffic
TRAFFIC_COUNT = 7
traffic = []                # dicts: ent, boxes, x, z, v0, v, label


def spawn_traffic():
    lane = random.choice((-LANE, LANE))
    z = ego.z + random.uniform(60, 180)
    for c in traffic:                                   # keep cars well apart
        if abs(c["x"] - lane) < 0.1 and abs(c["z"] - z) < 15:
            return
    body, name = random.choice(CAR_COLORS)
    boxes = []
    ent = create_car(boxes, lane, z, body)
    traffic.append({"ent": ent, "boxes": boxes, "x": lane, "z": z,
                    "v0": random.uniform(5, 10), "v": 0, "label": name})


def update_traffic(dt):
    # 1) each car drives at its own speed, but never faster than the car ahead
    for c in traffic:
        c["v"] = c["v0"]
        for o in traffic:
            if o is not c and abs(o["x"] - c["x"]) < 0.1 and 0 < o["z"] - c["z"] < 10:
                c["v"] = min(c["v"], o["v0"])
        if abs(ego.x - c["x"]) < 1.9 and 0 < ego.z - c["z"] < 10:   # slow for the ego car too
            c["v"] = min(c["v"], max(ego_speed, 0))

    # 2) move (a car never drives into the ego car)
    for c in traffic:
        nz = c["z"] + c["v"] * dt
        if abs(c["x"] - ego.x) < EGO_HX + 0.95 and abs(nz - ego.z) < EGO_HZ + 1.25:
            continue
        c["z"] = nz
        c["ent"].z = nz

    # 3) remove cars that are far behind / far ahead, then refill
    for c in traffic[:]:
        if c["z"] < ego.z - BEHIND or c["z"] > ego.z + AHEAD + 50:
            destroy(c["ent"])
            traffic.remove(c)
    if len(traffic) < TRAFFIC_COUNT:
        spawn_traffic()


# ------------------------------------------------------ object registry
objs = []                   # every obstacle this frame: (x, z, hx, hz, label, speed)


def get_objects():
    out = [o for ch in chunks.values() for o in ch["obstacles"]]
    out += [(c["x"], c["z"], 0.95, 1.25, c["label"], c["v"]) for c in traffic]
    return out


def all_boxes():
    for ch in chunks.values():
        yield from ch["boxes"]
    for c in traffic:
        yield from c["boxes"]
    yield from ego_boxes


# --------------------------------------------------------- driving logic
def blocked(nx, nz):
    for ox, oz, hx, hz, _, _ in objs:
        if abs(nx - ox) < EGO_HX + hx and abs(nz - oz) < EGO_HZ + hz:
            return True
    return False


def find_lead(centres):
    """Closest object ahead inside the corridor of the given x centres.
    Returns (bumper gap, its speed, label) or None."""
    best = None
    for ox, oz, hx, hz, label, v in objs:
        if oz <= ego.z:
            continue
        if not any(abs(ox - cx) < EGO_HX + hx + 0.15 for cx in centres):
            continue
        gap = (oz - hz) - (ego.z + EGO_HZ)
        if best is None or gap < best[0]:
            best = (gap, v, label)
    return best


def lane_free(cx):
    for ox, oz, hx, hz, _, _ in objs:
        if abs(ox - cx) < EGO_HX + hx + 0.15 and ego.z - 10 < oz < ego.z + 30:
            return False
    return True


CRUISE = 18             # autopilot cruise speed (m/s)
SAFE_GAP = 16           # start braking inside this gap (m)
LANE_CHANGE_GAP = 24    # try to change lane inside this gap (m)


def run_autopilot(dt):
    """Returns the steering step dx; updates ego_speed and status."""
    global ego_speed, target_x, status
    changing = abs(ego.x - target_x) > 0.3
    lead = find_lead([ego.x, target_x])
    status = "CRUISING"

    # Slow car ahead and the other lane is free -> change lane
    if lead and not changing and lead[0] < LANE_CHANGE_GAP and lead[1] < CRUISE - 1:
        other = -target_x
        if lane_free(other):
            target_x = other
            changing = True
            lead = find_lead([ego.x, target_x])

    desired = CRUISE
    if lead:
        gap, lv, _ = lead
        if gap < 3:
            desired, status = 0, "EMERGENCY STOP"
        elif gap < SAFE_GAP:
            desired, status = clamp(lv + (gap - 6) * 0.8, 0, CRUISE), "BRAKING"
    if changing and status == "CRUISING":
        status = "LANE CHANGE"

    ego_speed += clamp(desired - ego_speed, -25 * dt, 6 * dt)
    return clamp(target_x - ego.x, -5 * dt, 5 * dt)


def drive():
    global ego_speed, status
    dt = time.dt
    if autopilot_on:
        dx = run_autopilot(dt)
    else:
        top = 30 if held_keys["shift"] else 15
        ego_speed = (held_keys["up arrow"] - held_keys["down arrow"]) * top
        dx = (held_keys["right arrow"] - held_keys["left arrow"]) * 6 * dt
        status = "Manual"

    nx = clamp(ego.x + dx, -6, 6)
    if not blocked(nx, ego.z):
        ego.x = nx
    nz = ego.z + ego_speed * dt
    if not blocked(ego.x, nz):
        ego.z = nz
    else:
        ego_speed = 0


def follow_camera():
    target = ego.position + CAM_OFFSET
    camera.position += (target - camera.position) * min(1, 6 * time.dt)


# -------------------------------------------------------------------- HUD
hud_texts = []


def hud(text, x, y, scale=1):
    t = Text(text=text, position=(x, y), origin=(-0.5, 0), scale=scale, color=color.black)
    hud_texts.append(t)
    return t


hud("AUTONOMOUS VEHICLE ENVIRONMENT", -0.85, 0.47, 1.4)
hud("Computer Graphics Concept: Z-Buffer", -0.85, 0.42, 1)
mode_text = hud("Mode: Normal view", -0.85, 0.37, 1)
driver_text = hud("", -0.85, 0.32, 1.1)
hud("Arrows: drive | SHIFT: faster | A: autopilot | D: depth map", -0.85, 0.27, 0.8)
trip_text = hud("", 0.45, 0.47, 1.1)
speed_text = hud("", 0.45, 0.42, 1.1)
hud("Green: Ground | Grey: Road | Yellow: Obstacle | Magenta: Your car", -0.85, -0.46, 0.9)
nearest_text = Text(text="", position=(-0.85, -0.40), origin=(-0.5, 0),
                    scale=1.2, color=color.lime)


def update_hud():
    trip_text.text = f"Distance: {int(ego.z + 10)} m"
    speed_text.text = f"Speed: {abs(ego_speed) * 3.6:.0f} km/h"
    driver_text.text = "Driver: AUTOPILOT - " + status if autopilot_on else "Driver: MANUAL"

    best_gap, best_label = None, ""
    for ox, oz, hx, hz, label, _ in objs:
        gx = max(0, abs(ego.x - ox) - (EGO_HX + hx))
        gz = max(0, abs(ego.z - oz) - (EGO_HZ + hz))
        gap = (gx * gx + gz * gz) ** 0.5
        if best_gap is None or gap < best_gap:
            best_gap, best_label = gap, label

    if best_gap is None or best_gap > 40:
        nearest_text.text = "Nearest object: none within 40 m"
        nearest_text.color = color.lime
        return
    nearest_text.text = f"Nearest object: {best_label} - {best_gap:.1f} m"
    nearest_text.color = color.red if best_gap < 2 else color.orange if best_gap < 5 else color.lime


# ------------------------------------------------------------ depth mode
depth_mode = False
NEAR, FAR = 5, 100          # depth range along the camera view direction


def shade_depth():
    """Colour every object by its depth: near = white, far = black."""
    f = camera.forward
    cp = camera.world_position
    for e in all_boxes():
        p = e.world_position - cp
        d = p.x * f.x + p.y * f.y + p.z * f.z        # z-depth along the view axis
        g = int((1 - clamp((d - NEAR) / (FAR - NEAR), 0, 1)) * 255)
        e.color = rgb255(g, g, g)


def set_depth_mode(on):
    global depth_mode
    depth_mode = on
    window.color = color.black if on else SKY
    for t in hud_texts:
        t.color = color.white if on else color.black
    mode_text.text = ("Mode: DEPTH MAP (Z-Buffer)   near = white, far = black"
                      if on else "Mode: Normal view")
    if not on:
        for e in all_boxes():
            e.color = e.base_color


# ------------------------------------------------------------ main loops
manage_chunks()                                  # build the starting world
for _ in range(80):                              # fill the road with traffic
    if len(traffic) >= TRAFFIC_COUNT:
        break
    spawn_traffic()


def update():
    global objs
    update_traffic(time.dt)
    objs = get_objects()
    drive()
    manage_chunks()
    follow_camera()
    update_hud()
    if depth_mode:
        shade_depth()


def input(key):
    global autopilot_on, target_x
    if key == "d":
        set_depth_mode(not depth_mode)
    elif key == "a":
        autopilot_on = not autopilot_on
        target_x = -LANE if ego.x < 0 else LANE          # nearest lane
    elif key == "escape":
        application.quit()


app.run()
