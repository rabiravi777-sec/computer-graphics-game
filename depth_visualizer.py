"""
Autonomous Vehicle Environment Depth Visualizer   (OPTIMIZED - no lag)
Computer Graphics concept: Z-Buffer

What was making it lag, and what was changed
  1. Every box used to be its own Entity and then Entity.combine() merged them.
     Creating hundreds of Entities + combine() in Python is VERY slow and caused
     big hitches every time a chunk or a car was created.
     -> Now a tiny MeshBuilder writes the vertices of all boxes straight into ONE
        Mesh. No Entities are created for the boxes at all (10-50x faster).
  2. Cars were rebuilt from scratch every time one was spawned.
     -> Car meshes are cached per colour and shared.
  3. Obstacle list was rebuilt from all chunks every frame.
     -> Cached; only rebuilt when chunks/traffic change.
  4. Chunk generation is spread out: at most one chunk per frame, and a slightly
     shorter view distance (AHEAD) so fewer chunks exist.

Controls
    Up / Down arrows : drive forward / backward  (hold SHIFT to go faster)
    Left / Right     : steer
    A                : toggle AUTOPILOT on / off
    D                : toggle Normal view <-> Depth map
    Esc              : quit
"""
from ursina import *
from ursina.shaders import unlit_shader
import math
import random

app = Ursina()

window.title = "Autonomous Vehicle Environment Depth Visualizer"
window.size = (1200, 800)
window.borderless = False
window.fps_counter.enabled = True


# ---------------------------------------------------------------- helpers
def rgb255(r, g, b):
    """Colour helper that works on both old and new Ursina versions."""
    return color.rgb32(r, g, b) if hasattr(color, "rgb32") else color.rgb(r, g, b)


SKY = rgb255(135, 190, 225)
window.color = SKY

GROUND_COLOR = rgb255(50, 150, 60)
ROAD_COLOR = rgb255(65, 65, 70)
SIDEWALK_COLOR = rgb255(170, 170, 165)
CURB_COLOR = rgb255(200, 200, 195)
TRUNK_COLOR = rgb255(100, 70, 40)
LEAF_DARK = rgb255(30, 110, 40)
LEAF_MID = rgb255(45, 130, 50)
LEAF_LIGHT = rgb255(60, 150, 55)
WINDOW_COLOR = rgb255(150, 205, 235)
ROOF_COLOR = rgb255(90, 80, 80)
DOOR_COLOR = rgb255(90, 60, 40)
BUILDING_COLORS = [rgb255(190, 170, 140), rgb255(130, 160, 200),
                   rgb255(200, 120, 110), rgb255(160, 160, 170)]
CAR_COLORS = [(color.red, "Red car"), (color.blue, "Blue car"),
              (color.orange, "Orange car"), (color.white, "White car"),
              (color.gray, "Gray car")]
PARKED_COLORS = [rgb255(200, 200, 205), rgb255(40, 60, 120),
                 rgb255(150, 30, 30), rgb255(30, 30, 35), rgb255(220, 190, 60)]

# ------------------------------------------------------------ depth shader
NEAR, FAR = 5, 100
depth_mode = False

# gl_Position.w is the distance in front of the camera = the z-depth the
# Z-buffer works with. Near = white, far = black.
depth_shader = Shader(
    language=Shader.GLSL,
    vertex='''#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out float depth;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    depth = gl_Position.w;
}
''',
    fragment='''#version 140
uniform float near_d;
uniform float far_d;
in float depth;
out vec4 fragColor;
void main() {
    float g = 1.0 - clamp((depth - near_d) / (far_d - near_d), 0.0, 1.0);
    fragColor = vec4(g, g, g, 1.0);
}
''',
    default_input={"near_d": float(NEAR), "far_d": float(FAR)},
)


def current_shader():
    return depth_shader if depth_mode else unlit_shader


# ------------------------------------------------------------ mesh builder
# faces of a box as loops of corner indices; corner index = x*4 + y*2 + z (bits)
# (the bottom face is skipped - nobody ever sees it)
_FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
_CORNERS = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


class MeshBuilder:
    """Collects many boxes into one set of vertex arrays (=> ONE mesh)."""

    def __init__(self):
        self.verts = []
        self.tris = []
        self.cols = []

    def box(self, x, y, z, sx, sy, sz, col):
        base = len(self.verts)
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        verts, cols = self.verts, self.cols
        for cx, cy, cz in _CORNERS:
            verts.append((x + cx * hx, y + cy * hy, z + cz * hz))
            cols.append(col)
        tris = self.tris
        for a, b, c, d in _FACES:
            tris.append((base + a, base + b, base + c))
            tris.append((base + a, base + c, base + d))

    def mesh(self):
        return Mesh(vertices=self.verts, triangles=self.tris,
                    colors=self.cols, static=True)


def box(b, position, scale, color):
    """Same call style as before, but writes into a MeshBuilder."""
    b.box(position[0], position[1], position[2], scale[0], scale[1], scale[2], color)


def make_entity(builder, position=(0, 0, 0)):
    e = Entity(model=builder.mesh(), position=position, double_sided=True)
    e.shader = current_shader()
    return e


# ------------------------------------------------------- building blocks
def add_car_parts(b, x, z, body_color):
    """A car facing +z: body, cabin, wheels, headlights and tail lights."""
    box(b, position=(x, 0.5, z), scale=(1.7, 0.7, 2.5), color=body_color)
    box(b, position=(x, 1.05, z - 0.1), scale=(1.1, 0.5, 1.2), color=color.azure)
    for wx in (-0.9, 0.9):
        for wz in (-0.75, 0.75):
            box(b, position=(x + wx, 0.25, z + wz),
                scale=(0.25, 0.4, 0.5), color=color.black)
    for lx in (-0.55, 0.55):
        box(b, position=(x + lx, 0.55, z + 1.26),
            scale=(0.35, 0.2, 0.05), color=rgb255(255, 245, 190))
        box(b, position=(x + lx, 0.55, z - 1.26),
            scale=(0.35, 0.2, 0.05), color=rgb255(200, 20, 20))


_car_mesh_cache = {}


def create_car(x, z, body_color):
    """Car entity. The mesh is built once per colour and shared."""
    key = (round(body_color[0], 3), round(body_color[1], 3), round(body_color[2], 3))
    if key not in _car_mesh_cache:
        mb = MeshBuilder()
        add_car_parts(mb, 0, 0, body_color)
        _car_mesh_cache[key] = mb.mesh()
    e = Entity(model=_car_mesh_cache[key], position=(x, 0, z), double_sided=True)
    e.shader = current_shader()
    return e


def add_tree(b, x, z, s=1.0):
    box(b, position=(x, 1.0 * s, z), scale=(0.45 * s, 2.0 * s, 0.45 * s), color=TRUNK_COLOR)
    box(b, position=(x, 2.6 * s, z), scale=(2.6 * s, 1.5 * s, 2.6 * s), color=LEAF_DARK)
    box(b, position=(x, 3.8 * s, z), scale=(1.9 * s, 1.3 * s, 1.9 * s), color=LEAF_MID)
    box(b, position=(x, 4.8 * s, z), scale=(1.1 * s, 1.0 * s, 1.1 * s), color=LEAF_LIGHT)


def add_building(b, x, z, h, wall_color, side):
    w = 4
    box(b, position=(x, h / 2, z), scale=(w, h, w), color=wall_color)
    box(b, position=(x, h + 0.15, z), scale=(w + 0.3, 0.3, w + 0.3), color=ROOF_COLOR)
    face_x = x - side * (w / 2 + 0.03)
    rows = max(1, int((h - 2) / 2.2))
    for r in range(rows):
        for c in (-1.2, 0, 1.2):
            box(b, position=(face_x, 2.6 + r * 2.2, z + c),
                scale=(0.06, 1.0, 0.9), color=WINDOW_COLOR)
    box(b, position=(face_x, 0.9, z), scale=(0.08, 1.8, 1.0), color=DOOR_COLOR)


def add_shop(b, x, z):
    box(b, position=(x, 1.3, z), scale=(2.4, 2.6, 2.4), color=rgb255(225, 215, 190))
    box(b, position=(x, 2.75, z), scale=(2.9, 0.3, 2.9), color=rgb255(180, 40, 40))
    side = 1 if x > 0 else -1
    box(b, position=(x - side * 1.23, 1.2, z), scale=(0.06, 1.1, 1.6), color=WINDOW_COLOR)
    box(b, position=(x - side * 1.3, 2.3, z), scale=(0.3, 0.12, 2.6),
        color=rgb255(240, 240, 240))


# ------------------------------------------------------- infinite world
CHUNK = 25
AHEAD = 125         # was 160 -> fewer chunks alive at once
BEHIND = 45
LANE = 3.5
SHOULDER = 6.1
chunks = {}
chunks_changed = True


def build_chunk(i):
    """Create chunk number i as ONE mesh."""
    rng = random.Random(i * 7919 + 13)
    z0 = i * CHUNK
    b = MeshBuilder()
    obstacles = []

    box(b, position=(0, -0.6, 12.5), scale=(200, 1, CHUNK), color=GROUND_COLOR)
    box(b, position=(0, 0.05, 12.5), scale=(14, 0.2, CHUNK), color=ROAD_COLOR)
    for k in range(5):
        box(b, position=(0, 0.16, k * 5 + 2.5), scale=(0.15, 0.02, 2.5), color=color.white)
    for side in (-1, 1):
        box(b, position=(side * 8.0, 0.12, 12.5), scale=(2.0, 0.3, CHUNK), color=SIDEWALK_COLOR)
        box(b, position=(side * 6.95, 0.14, 12.5), scale=(0.12, 0.3, CHUNK), color=CURB_COLOR)

    for lz in (4, 16):
        for side in (-1, 1):
            add_building(b, side * rng.uniform(11.5, 13), lz + rng.uniform(-2, 2),
                         rng.uniform(5, 12), rng.choice(BUILDING_COLORS), side)

    for side in (-1, 1):
        for _ in range(2):
            add_tree(b, side * 8.6, rng.uniform(1, 24), rng.uniform(0.9, 1.3))

    if i >= 0 and rng.random() < 0.55:
        side = rng.choice((-1, 1))
        lz = rng.uniform(6, 19)
        kind = rng.choice(("tree", "car", "shop"))
        ox = side * SHOULDER
        if kind == "tree":
            add_tree(b, ox, lz, 1.1)
            obstacles.append((ox, z0 + lz, 0.6, 0.6, "Tree", 0))
        elif kind == "car":
            add_car_parts(b, ox, lz, rng.choice(PARKED_COLORS))
            obstacles.append((ox, z0 + lz, 0.95, 1.25, "Parked car", 0))
        else:
            add_shop(b, side * 6.4, lz)
            obstacles.append((side * 6.4, z0 + lz, 1.2, 1.2, "Shop", 0))

    root = make_entity(b, position=(0, 0, z0))
    return {"root": root, "obstacles": obstacles}


def manage_chunks(limit=1):
    global chunks_changed
    first = math.floor((ego.z - BEHIND) / CHUNK)
    last = math.floor((ego.z + AHEAD) / CHUNK)
    built = 0
    for i in range(first, last + 1):
        if i not in chunks:
            if built >= limit:
                break
            chunks[i] = build_chunk(i)
            built += 1
            chunks_changed = True
    for i in list(chunks):
        if i < first or i > last:
            destroy(chunks.pop(i)["root"])
            chunks_changed = True


# ---------------------------------------------------------------- ego car
EGO_HX, EGO_HZ = 0.95, 1.25
ego = create_car(LANE, -10, color.magenta)
ego_speed = 0.0
autopilot_on = False
target_x = LANE
status = "Manual"

# ----------------------------------------------------------------- camera
CAM_OFFSET = Vec3(0, 14, -22)
camera.fov = 70
camera.rotation = (28, 0, 0)
camera.clip_plane_far = 250
camera.position = ego.position + CAM_OFFSET

# ---------------------------------------------------------------- traffic
TRAFFIC_COUNT = 7
traffic = []


def spawn_traffic():
    lane = random.choice((-LANE, LANE))
    z = ego.z + random.uniform(60, 120)
    for c in traffic:
        if abs(c["x"] - lane) < 0.1 and abs(c["z"] - z) < 15:
            return
    body, name = random.choice(CAR_COLORS)
    ent = create_car(lane, z, body)
    traffic.append({"ent": ent, "x": lane, "z": z,
                    "v0": random.uniform(5, 10), "v": 0, "label": name})


def update_traffic(dt):
    for c in traffic:
        c["v"] = c["v0"]
        for o in traffic:
            if o is not c and abs(o["x"] - c["x"]) < 0.1 and 0 < o["z"] - c["z"] < 10:
                c["v"] = min(c["v"], o["v0"])
        if abs(ego.x - c["x"]) < 1.9 and 0 < ego.z - c["z"] < 10:
            c["v"] = min(c["v"], max(ego_speed, 0))

    for c in traffic:
        nz = c["z"] + c["v"] * dt
        if abs(c["x"] - ego.x) < EGO_HX + 0.95 and abs(nz - ego.z) < EGO_HZ + 1.25:
            continue
        c["z"] = nz
        c["ent"].z = nz

    for c in traffic[:]:
        if c["z"] < ego.z - BEHIND or c["z"] > ego.z + AHEAD + 50:
            destroy(c["ent"])
            traffic.remove(c)
    if len(traffic) < TRAFFIC_COUNT:
        spawn_traffic()


# ------------------------------------------------------ object registry
objs = []
_static_objs = []


def get_objects():
    """Static obstacles are cached; only moving traffic is rebuilt each frame."""
    global _static_objs, chunks_changed
    if chunks_changed:
        _static_objs = [o for ch in chunks.values() for o in ch["obstacles"]]
        chunks_changed = False
    return _static_objs + [(c["x"], c["z"], 0.95, 1.25, c["label"], c["v"]) for c in traffic]


def scene_entities():
    for ch in chunks.values():
        yield ch["root"]
    for c in traffic:
        yield c["ent"]
    yield ego


# --------------------------------------------------------- driving logic
def blocked(nx, nz):
    for ox, oz, hx, hz, _, _ in objs:
        if abs(nx - ox) < EGO_HX + hx and abs(nz - oz) < EGO_HZ + hz:
            return True
    return False


def find_lead(centres):
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


CRUISE = 18
SAFE_GAP = 16
LANE_CHANGE_GAP = 24


def run_autopilot(dt):
    global ego_speed, target_x, status
    changing = abs(ego.x - target_x) > 0.3
    lead = find_lead([ego.x, target_x])
    status = "CRUISING"

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
hud("Obstacles: Trees, Parked cars, Shops | Traffic: moving cars | Magenta: Your car",
    -0.85, -0.46, 0.9)
nearest_text = Text(text="", position=(-0.85, -0.40), origin=(-0.5, 0),
                    scale=1.2, color=color.lime)

_hud_cache = {}


def set_text(t, s):
    """Only rebuild the text mesh when the string really changed (it is slow)."""
    if _hud_cache.get(t) != s:
        _hud_cache[t] = s
        t.text = s


def update_hud():
    set_text(trip_text, f"Distance: {int(ego.z + 10)} m")
    set_text(speed_text, f"Speed: {abs(ego_speed) * 3.6:.0f} km/h")
    set_text(driver_text, "Driver: AUTOPILOT - " + status if autopilot_on else "Driver: MANUAL")

    best_gap, best_label = None, ""
    for ox, oz, hx, hz, label, _ in objs:
        gx = max(0, abs(ego.x - ox) - (EGO_HX + hx))
        gz = max(0, abs(ego.z - oz) - (EGO_HZ + hz))
        gap = (gx * gx + gz * gz) ** 0.5
        if best_gap is None or gap < best_gap:
            best_gap, best_label = gap, label

    if best_gap is None or best_gap > 40:
        set_text(nearest_text, "Nearest object: none within 40 m")
        nearest_text.color = color.lime
        return
    # round the gap to 0.5 m so the text mesh is not rebuilt for tiny changes
    set_text(nearest_text, f"Nearest object: {best_label} - {round(best_gap * 2) / 2:.1f} m")
    nearest_text.color = color.red if best_gap < 2 else color.orange if best_gap < 5 else color.lime


# ------------------------------------------------------------ depth mode
def set_depth_mode(on):
    global depth_mode
    depth_mode = on
    window.color = color.black if on else SKY
    for t in hud_texts:
        t.color = color.white if on else color.black
    set_text(mode_text, "Mode: DEPTH MAP (Z-Buffer)   near = white, far = black"
             if on else "Mode: Normal view")
    shader = current_shader()
    for e in scene_entities():
        e.shader = shader


# ------------------------------------------------------------ main loops
manage_chunks(limit=999)
for _ in range(80):
    if len(traffic) >= TRAFFIC_COUNT:
        break
    spawn_traffic()

hud_timer = 0.0


def update():
    global objs, hud_timer
    update_traffic(time.dt)
    objs = get_objects()
    drive()
    manage_chunks()                              # at most one new chunk per frame
    follow_camera()
    hud_timer += time.dt
    if hud_timer >= 0.1:
        hud_timer = 0.0
        update_hud()


def input(key):
    global autopilot_on, target_x
    if key == "d":
        set_depth_mode(not depth_mode)
    elif key == "a":
        autopilot_on = not autopilot_on
        target_x = -LANE if ego.x < 0 else LANE
    elif key == "escape":
        application.quit()


app.run()
