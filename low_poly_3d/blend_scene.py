import bpy, math, sys, random, time, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(3)
ROOT = os.environ.get("REEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/frames"
os.makedirs(OUT, exist_ok=True)
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
F0 = int(args[0]) if args else 1
F1 = int(args[1]) if len(args) > 1 else F0
STEP = int(args[2]) if len(args) > 2 else 1
SAMPLES = int(args[3]) if len(args) > 3 else 24
RESX, RESY = (int(args[4]), int(args[5])) if len(args) > 5 else (540, 960)
NF = 120

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
try:
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
except Exception:
    pass
sc.cycles.max_bounces = 4
sc.cycles.diffuse_bounces = 3
sc.cycles.glossy_bounces = 2
sc.cycles.transmission_bounces = 2
sc.cycles.use_adaptive_sampling = True
sc.render.resolution_x, sc.render.resolution_y = RESX, RESY
sc.render.resolution_percentage = 100
sc.render.fps = 24
sc.frame_start, sc.frame_end = 1, NF
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
sc.view_settings.exposure = 0.35

w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.02, 0.03, 0.05, 1); bg.inputs[1].default_value = 1.0


def mat(name, color, rough=0.7, emit=None, estr=0.0, bump=0.0, bscale=40.0, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = estr
    if bump > 0:
        n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = bscale; n.inputs["Detail"].default_value = 6
        bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump
        nt.links.new(n.outputs["Fac"], bp.inputs["Height"]); nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def img_emit_mat(name, path, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(path)
    nt.links.new(tx.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = 0.25
    return m


def put(obj, m, smooth=False):
    obj.data.materials.append(m)
    for p in obj.data.polygons: p.use_smooth = smooth
    return obj


def cube(name, loc, size, m, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name; o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    return put(o, m)


def cyl(name, loc, r, h, m, verts=8, rot=(0, 0, 0), r2=None):
    if r2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=loc, rotation=rot)
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r, radius2=r2, depth=h, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    return put(o, m)


def ico(name, loc, r, m, sub=1, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub, radius=r, location=loc)
    o = bpy.context.active_object; o.name = name; o.scale = scale
    return put(o, m)


def plane(name, loc, size, m, rot=(math.radians(90), 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name; o.scale = (size[0], size[1], 1)
    return put(o, m)


# ---------- generated images ----------
def font(sz, bold=True):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def stadium(w_, h_):
    im = Image.new("RGB", (640, 480)); d = ImageDraw.Draw(im)
    for i in range(12):
        t = i / 11
        d.rectangle([0, i * 14, 640, (i + 1) * 14], fill=(int(255 * (1 - t) + 80 * t), int(165 * (1 - t) + 35 * t), int(70 * (1 - t) + 70 * t)))
    d.rectangle([0, 150, 640, 270], fill=(60, 22, 30))
    for _ in range(300):
        x = random.randint(0, 635); y = random.randint(155, 262)
        d.rectangle([x, y, x + 5, y + 5], fill=random.choice([(255, 190, 90), (230, 90, 70), (250, 230, 190), (120, 50, 50)]))
    for x0 in (40, 600):
        d.rectangle([x0 - 8, 110, x0 + 8, 270], fill=(40, 20, 28)); d.rectangle([x0 - 38, 60, x0 + 38, 110], fill=(255, 240, 200))
    for i in range(6):
        d.rectangle([0, 270 + i * 35, 640, 270 + (i + 1) * 35], fill=(60, 140, 75) if i % 2 == 0 else (80, 165, 92))
    d.line([320, 270, 320, 480], fill=(240, 240, 230), width=5)
    d.ellipse([220, 340, 420, 440], outline=(240, 240, 230), width=4)
    d.ellipse([260, 60, 380, 150], fill=(255, 215, 130))
    return im.resize((w_, h_))


import numpy as np


def make_ui(clean):
    ui = Image.new("RGB", (1280, 720), (24, 24, 30)); d = ImageDraw.Draw(ui)
    d.rectangle([0, 0, 1280, 52], fill=(36, 36, 44))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([22 + i * 30, 18, 38 + i * 30, 34], fill=c)
    d.text((640, 26), "Photo Editor  -  DAY 1", font=font(22), fill=(200, 200, 210), anchor="mm")
    ph = stadium(780, 440)
    if not clean:
        a = np.array(ph).astype(float) * 0.62
        a += np.random.normal(0, 38, a.shape)
        ph = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    ui.paste(ph, (30, 72))
    d.text((40, 520), "IMG_0412.jpg", font=font(20), fill=(170, 170, 185))
    d.rectangle([830, 72, 1250, 512], fill=(34, 34, 42))
    for i, (nm, v) in enumerate([("Exposure", 0.25 if not clean else 0.72), ("Noise Reduction", 0.0 if not clean else 0.62), ("Contrast", 0.5), ("Highlights", 0.5), ("Shadows", 0.4)]):
        y = 110 + i * 78
        d.text((856, y), nm, font=font(22), fill=(210, 210, 222))
        d.rectangle([856, y + 38, 1220, y + 44], fill=(70, 72, 90))
        d.ellipse([856 + v * 364 - 12, y + 29, 856 + v * 364 + 12, y + 53], fill=(255, 150, 60) if i < 2 else (200, 200, 210))
    d.rectangle([30, 540, 1250, 700], fill=(30, 30, 38))
    cols = [(230, 120, 60), (60, 150, 160), (230, 190, 80), (170, 70, 70)]
    rnd = random.Random(5)
    for t in range(3):
        x = 40 + t * 18
        while x < 1100:
            wd = rnd.randint(90, 230)
            d.rectangle([x, 552 + t * 46, min(x + wd, 1240), 590 + t * 46], fill=rnd.choice(cols))
            x += wd + 8
    return ui


np.random.seed(2)
make_ui(False).save(f"{ROOT}/screen_grainy.png")
make_ui(True).save(f"{ROOT}/screen_clean.png")

sky = Image.new("RGB", (400, 520))
sd = ImageDraw.Draw(sky)
for y in range(520):
    t = y / 519
    sd.line([0, y, 400, y], fill=(int(55 + 125 * t), int(125 + 90 * t), int(215 + 30 * t)))
cl = Image.new("RGB", (400, 520), (0, 0, 0)); cd = ImageDraw.Draw(cl)
for (cx, cy, rx, ry) in [(100, 150, 110, 38), (230, 190, 130, 46), (320, 100, 90, 30), (150, 330, 150, 48), (300, 400, 120, 40), (60, 450, 90, 32), (260, 280, 70, 26)]:
    for k in range(7):
        ox = random.randint(-rx // 2, rx // 2); oy = random.randint(-ry // 3, ry // 3)
        cd.ellipse([cx + ox - rx // 2, cy + oy - ry // 2, cx + ox + rx // 2, cy + oy + ry // 2], fill=(255, 255, 255))
cl = cl.filter(ImageFilter.GaussianBlur(9))
sky = Image.composite(Image.new("RGB", (400, 520), (250, 250, 255)), sky, cl.convert("L").point(lambda v: int(v * 0.9)))
sky.save(f"{ROOT}/sky.png")

# ---------- materials ----------
M_WALL = mat("wall", (0.30, 0.05, 0.06), 0.95, bump=0.25, bscale=30)
M_FLOOR = mat("floor", (0.05, 0.025, 0.02), 0.85, bump=0.1, bscale=60)
M_DESK = mat("desk", (0.2, 0.1, 0.055), 0.6, bump=0.12, bscale=50)
M_ALU = mat("alu", (0.62, 0.63, 0.66), 0.28, metal=0.9)
M_DARK = mat("dark", (0.025, 0.022, 0.025), 0.4)
M_KEY = mat("kb", (0.8, 0.8, 0.82), 0.5)
M_HOOD = mat("hood", (0.07, 0.17, 0.16), 0.95, bump=0.6, bscale=220)
M_SKIN = mat("skin", (0.45, 0.28, 0.18), 0.8)
M_CAP = mat("cap", (0.55, 0.06, 0.05), 0.85, bump=0.4, bscale=200)
M_JEAN = mat("jean", (0.05, 0.07, 0.16), 0.9, bump=0.4, bscale=200)
M_CHAIR = mat("chair", (0.05, 0.05, 0.06), 0.6)
M_ROBOT = mat("robot", (0.65, 0.16, 0.05), 0.35, metal=0.2)
M_VISOR = mat("visor", (0.005, 0.005, 0.008), 0.15)
M_EYE = mat("eye", (1, 0.85, 0.5), 0.5, emit=(1, 0.8, 0.4), estr=14)
M_GOLD = mat("gold", (0.9, 0.6, 0.1), 0.25, metal=1.0)
M_LAMP = mat("lampshade", (1, 0.6, 0.25), 0.8, emit=(1, 0.55, 0.2), estr=3)
M_RING = mat("ring", (0.9, 0.9, 0.95), 0.2, metal=0.6)
M_FRAME = mat("wframe", (0.85, 0.82, 0.76), 0.6)
M_PENN = [mat("p%d" % i, c, 0.9) for i, c in enumerate([(0.9, 0.55, 0.1), (0.1, 0.45, 0.45), (0.8, 0.8, 0.7)])]
def screen_mat():
    m = bpy.data.materials.new("screen"); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    t1 = nt.nodes.new("ShaderNodeTexImage"); t1.image = bpy.data.images.load(f"{ROOT}/screen_grainy.png")
    t2 = nt.nodes.new("ShaderNodeTexImage"); t2.image = bpy.data.images.load(f"{ROOT}/screen_clean.png")
    mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
    rin = [i for i in mx.inputs if i.type == "RGBA"]; rout = [o for o in mx.outputs if o.type == "RGBA"][0]
    nt.links.new(t1.outputs["Color"], rin[0]); nt.links.new(t2.outputs["Color"], rin[1])
    nt.links.new(rout, b.inputs["Emission Color"]); nt.links.new(rout, b.inputs["Base Color"])
    b.inputs["Emission Strength"].default_value = 1.6; b.inputs["Roughness"].default_value = 0.25
    for f, v in ((1, 0.0), (100, 0.0), (112, 1.0), (NF, 1.0)):
        mx.inputs[0].default_value = v; mx.inputs[0].keyframe_insert("default_value", frame=f)
    return m


M_SCREEN = screen_mat()
M_SKY = img_emit_mat("sky", f"{ROOT}/sky.png", 2.6)

# ---------- set ----------
DZ = 1.0
cube("floor", (0, 0, -0.05), (14, 12, 0.1), M_FLOOR)
cube("wall", (0, 3.6, 6.0), (16, 0.2, 14), M_WALL)
cube("sidewall", (-6.0, 0, 6.0), (0.2, 12, 14), M_WALL)
cube("desk", (0.3, 1.2, DZ - 0.08), (5.2, 2.6, 0.16), M_DESK)
for x in (-2.1, 2.7):
    for y in (0.1, 2.3):
        cube("leg", (x, y, DZ / 2 - 0.08), (0.18, 0.18, DZ - 0.16), M_DESK)

# window with cloudy sky (right side of back wall)
WX, WZ = -2.7, 3.6
plane("sky", (WX, 3.47, WZ), (1.9, 2.5), M_SKY)
for (cx, cz, sx, sz) in [(WX, WZ + 1.3, 2.1, 0.14), (WX, WZ - 1.3, 2.1, 0.14), (WX - 1.0, WZ, 0.14, 2.7), (WX + 1.0, WZ, 0.14, 2.7), (WX, WZ, 0.07, 2.5), (WX, WZ + 0.1, 1.9, 0.07)]:
    cube("wf", (cx, 3.42 + (0.02 if sx > sz else 0.0) + (0.01 if sx == 0.07 or sz == 0.07 else 0.0), cz), (sx, 0.14, sz), M_FRAME)
cube("sill", (WX, 3.2, WZ - 1.4), (2.4, 0.5, 0.1), M_FRAME)

# computer monitor (Mac-style), lowered to eye level
MX, MY = 0.4, 2.0
cube("mon_panel", (MX, MY, DZ + 1.05), (2.5, 0.1, 1.5), M_ALU)
cube("mon_back", (MX, MY + 0.08, DZ + 1.1), (1.6, 0.14, 1.0), M_ALU)
cube("mon_neck", (MX, MY + 0.1, DZ + 0.2), (0.45, 0.14, 0.3), M_ALU)
cube("mon_base", (MX, MY - 0.05, DZ + 0.04), (1.1, 0.75, 0.08), M_ALU)
plane("screen", (MX, MY - 0.056, DZ + 1.12), (2.3, 1.3), M_SCREEN)
KX, KY = -0.15, 0.5
cube("keyboard", (KX, KY, DZ + 0.035), (1.2, 0.4, 0.07), M_KEY)
for r in range(3):
    cube("keys", (KX, KY - 0.12 + r * 0.12, DZ + 0.075), (1.1, 0.09, 0.03), M_DARK)
cube("mouse", (KX + 0.95, KY + 0.05, DZ + 0.05), (0.2, 0.3, 0.09), M_KEY)

# shelf: trophy + camera + pennants
cube("shelf", (1.7, 3.35, 4.0), (3.8, 0.5, 0.12), M_DESK)
cyl("tbase", (2.8, 3.35, 4.15), 0.3, 0.15, M_DESK, 8)
cyl("tstem", (2.8, 3.35, 4.35), 0.08, 0.3, M_GOLD, 8)
cyl("tcup", (2.8, 3.35, 4.75), 0.38, 0.5, M_GOLD, 8, r2=0.17)
ico("tball", (2.8, 3.35, 5.1), 0.17, M_GOLD)
cube("cam_body", (1.1, 3.35, 4.3), (0.9, 0.5, 0.45), M_DARK)
cyl("cam_lens", (1.1, 3.0, 4.3), 0.2, 0.25, M_RING, 8, rot=(math.radians(90), 0, 0))
for i, x in enumerate((0.3, 1.5, 2.7)):
    off = 0.5 if i % 2 else 0.0
    verts = [(x - 0.35, 3.48, 5.9 + off), (x + 0.35, 3.48, 5.9 + off), (x, 3.48, 5.1 + off)]
    me = bpy.data.meshes.new("pen"); me.from_pydata(verts, [], [(0, 1, 2)]); me.update()
    o = bpy.data.objects.new("pen", me); sc.collection.objects.link(o); o.data.materials.append(M_PENN[i % 3])

# lamp (left)
cyl("lbase", (2.7, 2.2, DZ + 0.05), 0.35, 0.1, M_DARK, 8)
cyl("lstem", (2.7, 2.2, DZ + 0.95), 0.06, 1.8, M_DARK, 6)
cyl("lshade", (2.7, 2.2, DZ + 1.9), 0.55, 0.6, M_LAMP, 8, r2=0.2)

# Claude: sits on top of the monitor, legs hanging over the front
TOP = DZ + 1.05 + 0.75
RBX = MX + 0.35
RB = (RBX, MY, TOP + 0.3)
M_CL = M_ROBOT
robot_root = bpy.data.objects.new("claude_root", None); sc.collection.objects.link(robot_root); robot_root.location = RB
parts = []
parts.append(ico("cbody", RB, 0.32, M_CL, sub=2, scale=(1, 0.9, 1)))
parts.append(cyl("cvisor", (RB[0], RB[1] - 0.27, RB[2] + 0.02), 0.2, 0.05, M_VISOR, 14, rot=(math.radians(90), 0, 0)))
eyeL = cube("eyeL", (RB[0] - 0.07, RB[1] - 0.31, RB[2] + 0.06), (0.06, 0.02, 0.05), M_EYE)
eyeR = cube("eyeR", (RB[0] + 0.07, RB[1] - 0.31, RB[2] + 0.06), (0.06, 0.02, 0.05), M_EYE)
mouth = cube("mouth", (RB[0], RB[1] - 0.31, RB[2] - 0.05), (0.13, 0.02, 0.025), M_EYE)
parts += [eyeL, eyeR, mouth,
          cyl("cant", (RB[0] + 0.04, RB[1], RB[2] + 0.4), 0.012, 0.2, M_RING, 6),
          ico("ctip", (RB[0] + 0.04, RB[1], RB[2] + 0.52), 0.05, M_EYE)]
for sx in (-1, 1):
    lx = RBX + sx * 0.14
    parts.append(cube("thigh", (lx, MY - 0.14, TOP + 0.07), (0.13, 0.34, 0.13), M_CL))
    parts.append(cube("shin", (lx, MY - 0.26, TOP - 0.17), (0.12, 0.12, 0.4), M_CL))
    parts.append(cube("foot", (lx, MY - 0.32, TOP - 0.4), (0.15, 0.24, 0.09), M_DARK))
bpy.context.view_layer.update()
for p in parts:
    p.parent = robot_root; p.matrix_parent_inverse = robot_root.matrix_world.inverted()
arm_piv = []
for sx in (-1, 1):
    pv = bpy.data.objects.new("armpiv", None); sc.collection.objects.link(pv); pv.location = (RBX + sx * 0.33, MY, RB[2] + 0.05)
    bpy.context.view_layer.update()
    pv.parent = robot_root; pv.matrix_parent_inverse = robot_root.matrix_world.inverted()
    arm = cube("arm", (RBX + sx * 0.33, MY, RB[2] + 0.05 - 0.14), (0.09, 0.09, 0.3), M_CL)
    bpy.context.view_layer.update()
    arm.parent = pv; arm.matrix_parent_inverse = pv.matrix_world.inverted()
    arm_piv.append(pv)
robot = [robot_root]

# character: faceless, back to camera, typing (no head scratching)
CH = (-0.15, -0.45)
cube("seat", (CH[0], CH[1], 0.62), (1.0, 1.0, 0.14), M_CHAIR)
cube("chairback", (CH[0], CH[1] - 0.55, 1.2), (1.0, 0.14, 1.0), M_CHAIR)
cyl("cstem", (CH[0], CH[1], 0.32), 0.07, 0.6, M_CHAIR, 6)
for k in range(5):
    a = k * 2 * math.pi / 5
    cube("cleg", (CH[0] + 0.35 * math.cos(a), CH[1] + 0.35 * math.sin(a), 0.04), (0.5, 0.07, 0.07), M_CHAIR, rot=(0, 0, a))
cube("torso", (CH[0], CH[1] - 0.02, 1.25), (0.95, 0.5, 1.05), M_HOOD)
cube("shoulders", (CH[0], CH[1] - 0.02, 1.7), (1.1, 0.5, 0.25), M_HOOD)
head_parts = [ico("head", (CH[0], CH[1] + 0.02, 2.12), 0.36, M_SKIN, sub=1),
              ico("cap", (CH[0], CH[1], 2.2), 0.4, M_CAP, sub=1, scale=(1, 1, 0.75)),
              cube("brim", (CH[0], CH[1] + 0.38, 2.15), (0.6, 0.35, 0.05), M_CAP)]
head_piv = bpy.data.objects.new("headpiv", None); sc.collection.objects.link(head_piv); head_piv.location = (CH[0], CH[1], 1.9)
bpy.context.view_layer.update()
for p in head_parts:
    p.parent = head_piv; p.matrix_parent_inverse = head_piv.matrix_world.inverted()
hands = []
for s in (-1, 1):
    cube("arm", (CH[0] + s * 0.52, CH[1] + 0.42, 1.36), (0.26, 0.95, 0.26), M_HOOD, rot=(math.radians(-14), 0, 0))
    hands.append(cube("hand", (CH[0] + s * 0.3, CH[1] + 0.92, DZ + 0.16), (0.22, 0.28, 0.14), M_SKIN))
    cube("thigh", (CH[0] + s * 0.28, CH[1] + 0.35, 0.78), (0.38, 0.95, 0.3), M_JEAN)
    cube("shin", (CH[0] + s * 0.28, CH[1] + 0.8, 0.35), (0.34, 0.3, 0.8), M_JEAN)
    cube("shoe", (CH[0] + s * 0.28, CH[1] + 0.95, 0.07), (0.38, 0.6, 0.14), M_DARK)

# coffee cup on desk
cyl("cup", (-1.1, 1.0, DZ + 0.2), 0.17, 0.4, M_PENN[1], 8)


# ---------- lights ----------
def light(name, kind, loc, energy, color, size=1.0, spot=None, aim_at=None):
    ld = bpy.data.lights.new(name, kind); ld.energy = energy; ld.color = color
    if kind == "AREA": ld.size = size
    if kind == "SPOT":
        ld.spot_size = math.radians(spot or 50); ld.spot_blend = 0.5; ld.shadow_soft_size = 0.2
    o = bpy.data.objects.new(name, ld); sc.collection.objects.link(o); o.location = loc
    if aim_at:
        e = bpy.data.objects.new("aim_" + name, None); e.location = aim_at; sc.collection.objects.link(e)
        c = o.constraints.new("TRACK_TO"); c.target = e; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return o


light("window", "AREA", (WX, 3.2, WZ), 900, (0.72, 0.85, 1.0), size=2.2, aim_at=(0.0, -0.5, 1.5))
light("screenglow", "AREA", (MX, MY - 0.9, DZ + 1.6), 140, (0.75, 0.82, 1.0), size=2.0, aim_at=(MX, -2.0, DZ + 1.2))
light("lamp", "POINT", (2.7, 2.0, DZ + 1.7), 380, (1, 0.6, 0.25))
light("fill", "AREA", (-3.0, -5.0, 4.5), 500, (1, 0.72, 0.5), size=3, aim_at=(0.0, 1.0, 1.6))
light("shelfspot", "SPOT", (1.7, -0.5, 6.2), 900, (1, 0.8, 0.5), spot=45, aim_at=(1.7, 3.3, 4.3))

# ---------- camera: starts on the right, drifts in toward the desk ----------
cam_d = bpy.data.cameras.new("cam"); cam_d.lens = 32
cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)
tc = cam.constraints.new("TRACK_TO"); tc.target = tgt; tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
for f, loc, t in [(1, (4.3, -6.6, 3.6), (0.7, 1.7, 2.4)),
                  (NF, (1.9, -4.7, 3.0), (0.1, 1.7, 2.3))]:
    cam.location = loc; cam.keyframe_insert("location", frame=f)
    tgt.location = t; tgt.keyframe_insert("location", frame=f)

# animation: Claude bob / blink / talking, guy head, typing
def kf(o, path, idx, f, v):
    setattr_target = getattr(o, path)
    setattr_target[idx] = v
    o.keyframe_insert(path, index=idx, frame=f)


for f in range(1, NF + 1, 6):
    kf(robot_root, "location", 2, f, RB[2] + 0.03 * math.sin(f / (NF - 1) * 4 * math.pi))
GUY = (4, 40)      # frames guy talks
CL1 = (46, 72)     # Claude line 1
CL2 = (75, 120)    # Claude line 2
for f in range(1, NF + 1, 2):
    talking = CL1[0] <= f <= CL1[1] or CL2[0] <= f <= CL2[1]
    kf(mouth, "scale", 2, f, 0.0125 * (3.2 if (talking and (f // 2) % 2 == 0) else 1.0))
for o in (eyeL, eyeR):
    for f, v in ((1, 1), (30, 1), (33, 0.1), (36, 1), (96, 1), (99, 0.1), (102, 1)):
        kf(o, "scale", 2, f, 0.025 * v)
# Claude waves right arm while talking
for f in range(1, NF + 1, 2):
    talking = CL1[0] <= f <= CL1[1] or CL2[0] <= f <= CL2[1]
    kf(arm_piv[1], "rotation_euler", 1, f, (math.radians(-35) + math.radians(18) * math.sin(f * 0.8)) if talking else 0.0)
# guy head: frustrated nod/shake while he talks, then looks up
for f, rx, rz in ((1, 0.0, 0.0), (8, 0.18, 0.0), (16, 0.2, 0.2), (22, 0.2, -0.2), (28, 0.2, 0.15), (40, 0.18, 0.0),
                  (60, 0.1, 0.0), (80, -0.05, 0.0), (120, -0.1, 0.0)):
    head_piv.rotation_euler = (rx, 0.0, rz); head_piv.keyframe_insert("rotation_euler", frame=f)
# typing hands
for i, h in enumerate(hands):
    z0 = h.location.z
    for f in range(1, NF + 1, 3):
        kf(h, "location", 2, f, z0 + (0.03 if (f // 3 + i) % 2 == 0 else 0.0))
    h.location.z = z0

# ---------- render ----------
t0 = time.time()
for f in range(F0, F1 + 1, STEP):
    path = f"{OUT}/f_{f:04d}.png"
    if os.path.exists(path):
        continue
    sc.frame_set(f)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("frame", f, round(time.time() - t0, 1), flush=True)
