"""The room (HQ look): wall shelf, window, desk, desk props, console, mouse pad, 60% keyboard, mouse, rounded CRT body.

Everything is procedural (PIL / numpy sprites placed in the 3D scene as textured quads) and lit by sampling the baked
wall / desk textures in cine.py, so props pick up the same window-blind light.  Layout constants are at the top of the
"layout" section; change them to move things.  cine.render() calls draw_room_back / draw_desk_set /
draw_monitor_body_hq / draw_desk_front when CINE_HQ=1.
"""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw
import cine as C

# ------------------------------------------------------------------ HQ room: desk props, console, shelf (all procedural)
_LW = np.array([0.299, 0.587, 0.114], np.float32)
_LM = {}


def _light_ratio(tex, x0, x1, p0, p1, w, h, ppu_x, ppu_y, ox, oy_top):
    """Local light level of a baked wall/desk texture over a rectangle, normalised by its median; resized to (h, w)."""
    key = id(tex)
    if key not in _LM:
        l = tex.astype(np.float32) @ _LW
        _LM[key] = (cv2.GaussianBlur(l, (0, 0), 5), float(np.median(l)))
    l, med = _LM[key]
    xa, xb = int((x0 - ox) * ppu_x), int((x1 - ox) * ppu_x)
    ya, yb = int((p0 - oy_top) * ppu_y), int((p1 - oy_top) * ppu_y)
    xa, ya = max(xa, 0), max(ya, 0)
    xb, yb = min(max(xb, xa + 2), l.shape[1]), min(max(yb, ya + 2), l.shape[0])
    return np.clip(cv2.resize(l[ya:yb, xa:xb] / med, (w, h)), 0.35, 1.5)


def wall_light(x0, x1, ytop, ybot, w, h):   # C.WALL: 6 px/unit, x from -110, y from 260 downward
    return _light_ratio(C.WALL, x0, x1, 260 - ytop, 260 - ybot, w, h, 6, 6, -110, 0)


def desk_light(x0, x1, z0, z1, w, h):       # C.DESK: 4 px/unit, x from -120, z from 0
    return _light_ratio(C.DESK, x0, x1, z0, z1, w, h, 4, 4, -120, 0)


def _lit(rgba, lm, tint=(1.0, 0.96, 0.90)):
    out = rgba.copy()
    out[:, :, :3] = np.clip(rgba[:, :, :3].astype(np.float32) * lm[:, :, None] * np.array(tint, np.float32), 0, 255).astype(np.uint8)
    return out


def _down(img_pil, f):
    a = np.array(img_pil)
    return cv2.resize(a, (a.shape[1] // f, a.shape[0] // f), interpolation=cv2.INTER_AREA)


def _lit_desk(rgba, x0, x1, z0, z1):
    return _lit(rgba, desk_light(x0, x1, z0, z1, rgba.shape[1], rgba.shape[0]))



def _lam(mk, depth_px, k=4.0, light=(-0.40, -0.60, 0.69)):
    """Lambert term from a dome height field built from a 0..1 mask (soft, solid-looking edges)."""
    hf = np.sqrt(np.clip(cv2.GaussianBlur(mk.astype(np.float32), (0, 0), depth_px), 0, 1))
    gy, gx = np.gradient(cv2.GaussianBlur(hf, (0, 0), max(depth_px * 0.12, 0.7)))
    kk = k * depth_px / 10.0
    nx, ny, nz = -gx * kk, -gy * kk, np.ones_like(gx)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    L = np.array(light, np.float32)
    return np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / nn, 0, 1)


def _paint(rgb, mk_img, color, depth_px, amb=0.40, gain=0.74, spec=0.0, k=4.0, rim=None):
    """Composite a shaded part into rgb (float HxWx3). mk_img: PIL 'L' mask."""
    mk = np.array(mk_img, np.float32) / 255.0
    lam = _lam(mk, depth_px, k)
    sh = amb + gain * lam
    col = np.array(color, np.float32)[None, None, :] * sh[:, :, None]
    if spec > 0:
        col = col + (np.clip((lam - 0.90) * 9, 0, 1) * spec)[:, :, None]
    if rim is not None:
        e = np.clip(mk - cv2.GaussianBlur(mk, (0, 0), max(depth_px * 0.35, 1.0)), 0, 1)
        col = col + e[:, :, None] * np.array(rim, np.float32)[None, None, :]
    a = cv2.GaussianBlur(mk, (0, 0), 0.6)[:, :, None]
    rgb[:] = rgb * (1 - a) + col * a
    return mk


def _mask(w, h, fn):
    m = Image.new('L', (w, h), 0)
    fn(ImageDraw.Draw(m))
    return m


def _blur_a(rgba, s):
    return np.dstack([cv2.GaussianBlur(rgba[:, :, :3], (0, 0), s), cv2.GaussianBlur(rgba[:, :, 3], (0, 0), s)])


def _soft_shadow(w_u, h_u, ppu, opacity, blur_u):
    pad = blur_u * 2.2
    w, h = int((w_u + 2 * pad) * ppu), int((h_u + 2 * pad) * ppu)
    m = np.zeros((h, w), np.float32)
    m[int(pad * ppu):h - int(pad * ppu), int(pad * ppu):w - int(pad * ppu)] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), blur_u * ppu * 0.5)
    t = np.zeros((h, w, 4), np.uint8)
    t[:, :, 3] = (m * 255 * opacity).astype(np.uint8)
    return t, pad


def _hq_prop(name):
    return _PROPS[name]


def make_mat():
    ppu = 20
    w_u, d_u = 62, 31
    tw, th = w_u * ppu, d_u * ppu
    rs = np.random.RandomState(21)
    base = np.array([44, 41, 46], np.float32)[None, None, :] * (1 + 0.05 * cv2.GaussianBlur(rs.randn(th, tw).astype(np.float32), (0, 0), 0.9))[:, :, None]
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    ins = 16
    for x in range(ins, tw - ins, 14):          # amber stitched border
        d.line([x, ins, x + 8, ins], fill=(176, 118, 54), width=2)
        d.line([x, th - ins, x + 8, th - ins], fill=(176, 118, 54), width=2)
    for y in range(ins, th - ins, 14):
        d.line([ins, y, ins, y + 8], fill=(176, 118, 54), width=2)
        d.line([tw - ins, y, tw - ins, y + 8], fill=(176, 118, 54), width=2)
    a = np.array(im, np.float32)
    lm = desk_light(-31, 31, 51, 82, tw, th)
    a *= lm[:, :, None] * np.array([1.0, 0.97, 0.92], np.float32)
    al = Image.new('L', (tw, th), 0)
    ImageDraw.Draw(al).rounded_rectangle([0, 0, tw - 1, th - 1], radius=60, fill=255)
    al = cv2.GaussianBlur(np.array(al), (0, 0), 1.2)
    return np.dstack([np.clip(a, 0, 255).astype(np.uint8), al])


KB60_ROWS = [
    [("ESC", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1), ("-", 1), ("=", 1), ("<-", 2)],
    [("TAB", 1.5)] + [(c, 1) for c in "QWERTYUIOP[]"] + [("\\", 1.5)],
    [("CAPS", 1.75)] + [(c, 1) for c in "ASDFGHJKL;'"] + [("ENTER", 2.25)],
    [("SHIFT", 2.25)] + [(c, 1) for c in "ZXCVBNM,./"] + [("SHIFT", 2.75)],
    [("CTRL", 1.25), ("WIN", 1.25), ("ALT", 1.25), ("", 6.25), ("ALT", 1.25), ("FN", 1.25), ("MENU", 1.25), ("CTRL", 1.25)],
]


def make_kb60():
    """60% keyboard (61 keys), cream retro case with two amber accent keys. 30 x 11.25 units.
    Built at high resolution: bevelled case, dished keycaps with side walls, soft key shadows, crisp legends."""
    S, u = 2, 150
    tw, th = 16 * u * S, int(6 * u * S)
    U = u * S
    rgb = np.zeros((th, tw, 3), np.float32)
    rgb[:] = (150, 137, 114)
    case = _mask(tw, th, lambda d: d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=int(0.36 * U), fill=255))
    _paint(rgb, case, (196, 182, 154), 0.20 * U, amb=0.55, gain=0.55, rim=(30, 26, 20))
    plate = _mask(tw, th, lambda d: d.rounded_rectangle([int(0.17 * U), int(0.17 * U), tw - int(0.17 * U), th - int(0.17 * U)], radius=int(0.25 * U), fill=255))
    _paint(rgb, plate, (120, 108, 88), 0.06 * U, amb=0.65, gain=0.30)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, 'RGBA')
    gap = 0.07 * U
    for r, row in enumerate(KB60_ROWS):
        x = 0.5 * U
        y = 0.5 * U + r * U
        for lab, wu in row:
            x0, y0, x1, y1 = x + gap, y + gap * 0.6, x + wu * U - gap, y + U - gap * 1.4
            accent = lab in ("ESC", "ENTER")
            top = (244, 166, 78) if accent else (232, 222, 198)
            edge = (206, 122, 40) if accent else (176, 163, 138)
            rr = 0.14 * U
            d.rounded_rectangle([x0 - 0.02 * U, y0 + 0.10 * U, x1 + 0.02 * U, y1 + 0.16 * U], rr, fill=(60, 50, 38, 90))   # contact shadow
            d.rounded_rectangle([x0, y0 + 0.06 * U, x1, y1 + 0.10 * U], rr, fill=edge + (255,))                          # side wall
            d.rounded_rectangle([x0, y0, x1, y1], rr, fill=tuple(int(c * 0.90) for c in top) + (255,))                    # cap edge
            for i in range(1, 7):                                                                                           # dished top
                f = i / 7.0
                ins = 0.012 * U * i * 1.4
                col = tuple(int(top[k] * (0.90 + 0.12 * f)) for k in range(3))
                d.rounded_rectangle([x0 + ins, y0 + ins * 0.9, x1 - ins, y1 - ins * 1.15], max(rr - ins, 2), fill=col + (255,))
            d.line([x0 + rr, y0 + 0.02 * U, x1 - rr, y0 + 0.02 * U], fill=(255, 250, 236, 150), width=max(int(0.018 * U), 1))   # top-edge highlight
            if lab:
                fs = int(0.30 * U) if len(lab) == 1 else int(0.20 * U if len(lab) > 3 else 0.23 * U)
                d.text(((x0 + x1) / 2, (y0 + y1) / 2 - 0.02 * U), lab, font=C.mono(fs), fill=(86, 74, 58, 255), anchor='mm')
            x += wu * U
    a = np.array(im, np.float32)
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    a *= (1.04 - 0.10 * (yy / th) + 0.03 * (1 - xx / tw))[:, :, None]
    al = np.array(case)
    a = cv2.resize(a, (tw // S, th // S), interpolation=cv2.INTER_AREA)
    al = cv2.resize(al, (tw // S, th // S), interpolation=cv2.INTER_AREA)
    a *= desk_light(-23, 7, 57, 68.25, a.shape[1], a.shape[0])[:, :, None] * np.array([1.0, 0.97, 0.92], np.float32)
    return np.dstack([np.clip(a, 0, 255).astype(np.uint8), al])


def make_mouse():
    """Cream gaming mouse seen from above. 4.4 x 7 units, high-res shaded shell, split buttons, ridged scroll wheel."""
    S, ppu = 2, 90
    W, H = 4.4, 7.0
    w, h = int(W * ppu * S), int(H * ppu * S)
    U = ppu * S
    rgb = np.zeros((h, w, 3), np.float32)
    shell = _mask(w, h, lambda d: d.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(w * 0.47), fill=255))
    _paint(rgb, shell, (232, 220, 194), 0.55 * U, amb=0.46, gain=0.68, spec=34, k=3.2, rim=(18, 14, 10))
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, 'RGBA')
    cx = w // 2
    d.line([cx, 0.3 * U, cx, 2.95 * U], fill=(112, 98, 78, 255), width=int(0.07 * U))                         # button split
    d.line([cx + 0.04 * U, 0.3 * U, cx + 0.04 * U, 2.95 * U], fill=(255, 250, 236, 110), width=int(0.03 * U))
    d.arc([0.35 * U, 0.0, w - 0.35 * U, 5.9 * U], 205, 335, fill=(128, 114, 92, 255), width=int(0.05 * U))    # (curve of button front)
    d.line([0.42 * U, 2.95 * U, w - 0.42 * U, 2.95 * U], fill=(138, 124, 100, 255), width=int(0.06 * U))       # button/palm seam
    d.line([0.42 * U, 2.99 * U, w - 0.42 * U, 2.99 * U], fill=(255, 250, 236, 100), width=int(0.025 * U))
    wl = _mask(w, h, lambda dd: dd.rounded_rectangle([cx - 0.19 * U, 0.85 * U, cx + 0.19 * U, 2.15 * U], int(0.16 * U), fill=255))
    rgb2 = np.array(im, np.float32)
    _paint(rgb2, wl, (58, 50, 44), 0.12 * U, amb=0.5, gain=0.8, spec=40)
    im = Image.fromarray(np.clip(rgb2, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, 'RGBA')
    for k in range(9):
        yy_ = 0.95 * U + k * 0.13 * U
        d.line([cx - 0.17 * U, yy_, cx + 0.17 * U, yy_], fill=(20, 18, 16, 190), width=max(int(0.035 * U), 1))
    d.ellipse([cx - 0.16 * U, 5.35 * U - 0.16 * U, cx + 0.16 * U, 5.35 * U + 0.16 * U], fill=(214, 140, 60, 255))   # amber DPI dot
    a = cv2.resize(np.array(im), (int(W * ppu), int(H * ppu)), interpolation=cv2.INTER_AREA)
    al = cv2.resize(np.array(shell), (int(W * ppu), int(H * ppu)), interpolation=cv2.INTER_AREA)
    lm = desk_light(17.8, 22.2, 61.5, 68.5, a.shape[1], a.shape[0])
    return _lit(np.dstack([a, al]), lm)


def make_console():
    """Generic white tower console on a small stand: two curved white wings around a black core, cool LED line.
    No logos. 12 x 34 units."""
    S, ppu = 3, 30
    w, h = 12 * ppu * S, 34 * ppu * S
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / w) * 2 - 1
    v = yy / h
    body_v = np.clip(v / 0.93, 0, 1)                                   # the tower itself occupies the top 93%
    half = 0.50 + 0.34 * (1 - body_v) ** 0.85 + 0.035 * np.sin(body_v * 5.5 * np.pi) * (1 - body_v)   # flaring, softly waved wings
    cap_t = np.clip(body_v / 0.055, 0, 1)
    cap_b = np.clip((1 - body_v) / 0.03, 0, 1)
    reach = half * np.sqrt(cap_t * cap_b)
    mask = ((np.abs(u) < reach) & (v < 0.93)).astype(np.float32)
    wing = np.abs(u) / np.maximum(half, 1e-3)
    shade = 1.0 - 0.26 * wing ** 1.8 - 0.07 * v + 0.12 * np.exp(-((u + 0.55 * half) / 0.16) ** 2)
    rgb = np.array([240, 242, 248], np.float32)[None, None, :] * shade[:, :, None]
    cw = 0.24 * (0.65 + 0.35 * (1 - body_v))
    core = (np.abs(u) < cw) & (body_v > 0.06) & (body_v < 0.96)
    cshade = 1.0 + 0.9 * np.exp(-((u + 0.06) / 0.06) ** 2) * (1 - body_v) * 0.6
    rgb = np.where(core[:, :, None], np.array([12, 12, 17], np.float32)[None, None, :] * cshade[:, :, None], rgb)
    led = np.exp(-(u / 0.025) ** 2) * ((body_v > 0.55) & (body_v < 0.90))
    glow = cv2.GaussianBlur(led.astype(np.float32), (0, 0), 7 * S)
    rgb = rgb + (led[:, :, None] * 1.0 + glow[:, :, None] * 0.7) * np.array([120, 175, 255], np.float32)[None, None, :]
    # stand: a slim white/dark disc under the tower
    foot = (v > 0.93) & (v < 0.985) & (np.abs(u) < 0.62 * np.sqrt(np.clip((0.985 - v) / 0.055, 0, 1)))
    rgb = np.where(foot[:, :, None], np.array([196, 198, 206], np.float32)[None, None, :] * (1.0 - 0.3 * (v[:, :, None] - 0.93) / 0.055), rgb)
    mask = np.maximum(mask, foot.astype(np.float32))
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2 * S)
    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return cv2.resize(rgba, (w // S, h // S), interpolation=cv2.INTER_AREA)


def make_plant():
    S, ppu = 3, 30
    w, h = 16 * ppu * S, 20 * ppu * S
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rs = np.random.RandomState(4)
    cx, soil = w // 2, h - int(5.4 * ppu * S)
    for k in range(13):
        ang = math.radians(-78 + k * 13 + rs.uniform(-5, 5))
        L = rs.uniform(8.0, 13.5) * ppu * S
        wd = rs.uniform(1.4, 2.2) * ppu * S
        bend = rs.uniform(-0.25, 0.25)
        pts_l, pts_r = [], []
        for t in np.linspace(0, 1, 18):
            a_ = ang + bend * t
            px, py = cx + math.sin(a_) * L * t * (1 - 0.15 * t), soil - math.cos(a_) * L * t
            nx, ny = math.cos(a_), math.sin(a_)
            hw = wd * math.sin(math.pi * min(t * 1.05, 1)) ** 0.85
            pts_l.append((px - nx * hw, py - ny * hw))
            pts_r.append((px + nx * hw, py + ny * hw))
        g = rs.uniform(0.8, 1.15)
        d.polygon(pts_l + pts_r[::-1], fill=(int(46 * g), int(104 * g), int(58 * g), 255))
        d.polygon(pts_l + [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in zip(pts_l[::-1], pts_r[::-1])][::-1][:0] + pts_l[::-1][:0], fill=None)
        mid = [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in zip(pts_l, pts_r)]
        d.line(mid, fill=(int(88 * g), int(150 * g), int(92 * g), 255), width=3 * S)
    top, bot = h - int(5.4 * ppu * S), h - 2
    tw_, bw_ = 6.6 * ppu * S, 4.8 * ppu * S
    d.polygon([(cx - tw_ / 2, top), (cx + tw_ / 2, top), (cx + bw_ / 2, bot), (cx - bw_ / 2, bot)], fill=(176, 94, 62, 255))
    d.rectangle([cx - tw_ / 2 - 6 * S, top - 8 * S, cx + tw_ / 2 + 6 * S, top + int(1.3 * ppu * S)], fill=(196, 108, 72, 255))
    d.polygon([(cx - tw_ / 2, top + int(1.3 * ppu * S)), (cx - tw_ / 2 + 14 * S, top + int(1.3 * ppu * S)), (cx - bw_ / 2 + 12 * S, bot), (cx - bw_ / 2, bot)], fill=(150, 78, 52, 255))
    return _down(im, S)


def make_lens():
    """A telephoto lens standing on its mount, seen from the side. 6.4 x 9.5 units."""
    S, ppu = 3, 30
    w, h = int(6.4 * ppu * S), int(9.5 * ppu * S)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    U = ppu * S
    cx = w / 2
    d.rectangle([cx - 2.7 * U, h - 0.45 * U, cx + 2.7 * U, h], fill=(170, 170, 178, 255))                          # silver mount
    d.rounded_rectangle([cx - 3.2 * U, h - 5.9 * U, cx + 3.2 * U, h - 0.45 * U], 0.3 * U, fill=(24, 24, 28, 255))   # main barrel
    for i in range(0, 26):                                                                                          # focus-ring ribs
        x = cx - 3.0 * U + i * 0.24 * U
        d.line([x, h - 5.2 * U, x, h - 1.4 * U], fill=(46, 46, 52, 255), width=S)
    d.rectangle([cx - 3.25 * U, h - 6.3 * U, cx + 3.25 * U, h - 5.9 * U], fill=(176, 176, 186, 255))               # silver ring
    d.rounded_rectangle([cx - 2.7 * U, h - 9.5 * U, cx + 2.7 * U, h - 6.3 * U], 0.35 * U, fill=(28, 28, 32, 255))   # upper barrel
    d.rectangle([cx - 2.7 * U, h - 9.5 * U, cx + 2.7 * U, h - 9.2 * U], fill=(60, 60, 68, 255))
    d.rectangle([cx - 3.0 * U, h - 2.2 * U, cx - 2.7 * U, h - 1.0 * U], fill=(230, 230, 236, 255))                  # distance mark
    d.line([cx - 2.9 * U, h - 5.6 * U, cx - 2.9 * U, h - 0.7 * U], fill=(96, 96, 106, 255), width=int(0.22 * U))     # left highlight
    d.line([cx - 2.4 * U, h - 9.3 * U, cx - 2.4 * U, h - 6.5 * U], fill=(88, 88, 98, 255), width=int(0.18 * U))
    return _down(im, S)


def make_books():
    S, ppu = 3, 30
    w, h = 11 * ppu * S, 5.2 * ppu * S
    im = Image.new('RGBA', (int(w), int(h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    U = ppu * S
    cols = [(150, 70, 50), (60, 78, 104), (196, 160, 90)]
    y = h
    for i, (cc, wu, ox) in enumerate(zip(cols, (10.0, 9.0, 8.4), (0.0, 0.7, 0.3))):
        th_ = 1.5 * U
        x0, x1 = ox * U, (ox + wu) * U
        d.rectangle([x0, y - th_, x1, y], fill=(226, 214, 188, 255))
        d.rectangle([x0, y - th_, x1, y - th_ + 0.28 * U], fill=cc + (255,))
        d.rectangle([x0, y - 0.28 * U, x1, y], fill=cc + (255,))
        d.rectangle([x0, y - th_, x0 + 0.5 * U, y], fill=tuple(int(c * 0.8) for c in cc) + (255,))
        y -= th_
    return _down(im, S)


def make_drawers():
    ppu = 12
    tw, th = 90 * ppu, int(57 * ppu)
    rs = np.random.RandomState(12)
    g = cv2.GaussianBlur(rs.randn(th, tw).astype(np.float32), (0, 0), sigmaX=60, sigmaY=0.8)
    g /= np.abs(g).max() + 1e-6
    base = np.array([84, 54, 34], np.float32)[None, None, :] * (1 + 0.18 * g)[:, :, None] * 0.55
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for (y0, y1) in ((10, 26), (30, 46)):
        d.rounded_rectangle([14 * ppu // 4, y0 * ppu, tw - 14 * ppu // 4, y1 * ppu], 8, outline=(40, 26, 16), width=4)
        d.rounded_rectangle([tw // 2 - 70, (y0 + y1) * ppu // 2 - 5, tw // 2 + 70, (y0 + y1) * ppu // 2 + 5], 5, fill=(150, 126, 92))
    a = np.dstack([np.array(im), np.full((th, tw), 255, np.uint8)])
    return a


def make_shelf_board():
    ppu = 20
    tw, th = 72 * ppu, int(2.6 * ppu)
    rs = np.random.RandomState(8)
    g = cv2.GaussianBlur(rs.randn(th, tw).astype(np.float32), (0, 0), sigmaX=40, sigmaY=0.7)
    g /= np.abs(g).max() + 1e-6
    base = np.array([124, 84, 54], np.float32)[None, None, :] * (1 + 0.22 * g)[:, :, None]
    base *= (1.12 - 0.35 * np.linspace(0, 1, th)[:, None, None])
    base[:2] *= 1.25
    a = np.dstack([np.clip(base, 0, 255).astype(np.uint8), np.full((th, tw), 255, np.uint8)])
    return _lit(a, wall_light(-36, 36, 181, 178.4, tw, th))


def make_camera_body():
    """Compact mirrorless camera with a short zoom lens, side view, lens pointing left. 19 x 10.5 units.
    Generic design, no brand marks. High-res: shaded rubber grip with leatherette, cylindrical lens, knurled dials."""
    S, ppu = 2, 70
    W, H = 19.0, 10.5
    w, h = int(W * ppu * S), int(H * ppu * S)
    U = ppu * S
    rs = np.random.RandomState(7)
    rgb = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # body
    body = _mask(w, h, lambda d: d.rounded_rectangle([6.2 * U, 3.0 * U, 18.4 * U, 10.4 * U], int(0.75 * U), fill=255))
    _paint(rgb, body, (44, 44, 52), 0.45 * U, amb=0.32, gain=0.95, spec=70, k=3.0, rim=(70, 54, 40))
    # viewfinder hump + flash hump
    hump = _mask(w, h, lambda d: d.rounded_rectangle([7.0 * U, 1.15 * U, 10.7 * U, 3.3 * U], int(0.55 * U), fill=255))
    _paint(rgb, hump, (42, 42, 50), 0.30 * U, amb=0.32, gain=0.95, spec=60, rim=(70, 54, 40))
    # rubber grip with leatherette texture
    grip = _mask(w, h, lambda d: d.rounded_rectangle([14.7 * U, 3.0 * U, 18.4 * U, 10.4 * U], int(0.8 * U), fill=255))
    gm = np.array(grip, np.float32) / 255
    tex = cv2.GaussianBlur(rs.randn(h, w).astype(np.float32), (0, 0), 0.0032 * U * 12)
    gl = _lam(gm, 0.5 * U, 3.4)
    grc = (np.array([30, 30, 36], np.float32)[None, None, :] * (0.40 + 0.95 * gl)[:, :, None]) * (1 + 0.22 * tex)[:, :, None]
    ga = cv2.GaussianBlur(gm, (0, 0), 0.6)[:, :, None]
    rgb[:] = rgb * (1 - ga) + grc * ga
    # shutter dial / mode dial
    sd = _mask(w, h, lambda d: d.rounded_rectangle([14.5 * U, 2.0 * U, 17.1 * U, 3.12 * U], int(0.14 * U), fill=255))
    _paint(rgb, sd, (176, 178, 188), 0.20 * U, amb=0.45, gain=0.80, spec=70, k=2.0)
    for k in range(26):
        x_ = 14.6 * U + k * 0.1 * U
        rgb[int(2.02 * U):int(3.1 * U), int(x_):int(x_ + 0.03 * U)] *= 0.55
    sb = _mask(w, h, lambda d: d.ellipse([17.1 * U, 2.3 * U, 18.1 * U, 3.2 * U], fill=255))
    _paint(rgb, sb, (214, 40, 36), 0.18 * U, amb=0.45, gain=0.8, spec=60)               # red record button
    hs = _mask(w, h, lambda d: d.rectangle([11.1 * U, 2.5 * U, 14.2 * U, 3.05 * U], fill=255))
    _paint(rgb, hs, (150, 152, 162), 0.08 * U, amb=0.55, gain=0.7, spec=40)             # hot shoe
    # lens: mount ring, barrel (cylinder shading), zoom ring ribs, front ring
    def cyl(x0, x1, y0, y1, col, gain=0.95, amb=0.30, spec=40):
        m = _mask(w, h, lambda d: d.rounded_rectangle([x0 * U, y0 * U, x1 * U, y1 * U], int(0.12 * U), fill=255))
        mk = np.array(m, np.float32) / 255
        t = np.clip((yy / U - y0) / (y1 - y0), 0, 1) * 2 - 1
        lam = np.clip(1 - ((t + 0.42) / 1.42) ** 2, 0, 1)
        shade = amb + gain * lam
        c = np.array(col, np.float32)[None, None, :] * shade[:, :, None]
        c += (np.clip((lam - 0.86) * 7, 0, 1) * spec)[:, :, None]
        a_ = cv2.GaussianBlur(mk, (0, 0), 0.6)[:, :, None]
        rgb[:] = rgb * (1 - a_) + c * a_
        return mk
    cyl(6.05, 6.7, 3.65, 9.85, (200, 202, 212), spec=70)                                  # silver mount ring
    barrel = cyl(1.2, 6.1, 3.7, 9.8, (46, 46, 54), spec=60)
    for k in range(0, 40):                                                                # zoom ring ribs
        x_ = (1.55 + k * 0.105) * U
        c0, c1 = int(x_), int(x_ + 0.05 * U)
        rgb[int(3.85 * U):int(9.65 * U), c0:c1] *= 0.55
    cyl(0.95, 1.25, 3.85, 9.65, (190, 192, 202), spec=60)                                 # focus/silver trim
    cyl(0.12, 1.0, 3.95, 9.55, (36, 36, 44), spec=50)                                     # lens front barrel
    cyl(0.04, 0.2, 4.1, 9.4, (24, 24, 30), spec=18)                                       # filter ring
    # side details: strap lug, rear panel texture, body seam
    lug = _mask(w, h, lambda d: d.rounded_rectangle([6.0 * U, 4.2 * U, 6.5 * U, 4.8 * U], int(0.1 * U), fill=255))
    _paint(rgb, lug, (160, 162, 172), 0.06 * U, amb=0.5, gain=0.7, spec=40)
    pass
    pass
    # ambient bounce: slightly warm and lifted at the bottom edge, so it doesn't vanish into the desk
    vg = np.clip(1.45 - 0.75 * ((yy / U - 1.0) / 9.4), 0.55, 1.5)[:, :, None]
    out = np.clip(rgb * vg * 1.25, 0, 255)
    al = np.maximum.reduce([np.array(m_, np.uint8) for m_ in (body, hump, grip, sd, sb, barrel)] + [np.array(_mask(w, h, lambda d: d.rounded_rectangle([0.04 * U, 3.7 * U, 6.7 * U, 9.85 * U], int(0.2 * U), fill=255)))])
    a_img = cv2.resize(out, (int(W * ppu), int(H * ppu)), interpolation=cv2.INTER_AREA).astype(np.uint8)
    al = cv2.resize(al, (int(W * ppu), int(H * ppu)), interpolation=cv2.INTER_AREA)
    return np.dstack([a_img, al])


def make_mug():
    S, ppu = 3, 40
    w, h = int(5.4 * ppu * S), int(4.6 * ppu * S)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    U = ppu * S
    d.ellipse([3.4 * U, 1.0 * U, 5.3 * U, 3.4 * U], outline=(226, 216, 192, 255), width=int(0.42 * U))   # handle
    d.rounded_rectangle([0.2 * U, 0.55 * U, 3.9 * U, 4.5 * U], 0.35 * U, fill=(230, 221, 198, 255))      # body
    d.rectangle([0.2 * U, 2.1 * U, 3.9 * U, 2.45 * U], fill=(214, 140, 60, 255))                          # amber band
    d.ellipse([0.2 * U, 0.1 * U, 3.9 * U, 1.1 * U], fill=(210, 198, 172, 255))                            # rim
    d.ellipse([0.42 * U, 0.2 * U, 3.68 * U, 0.98 * U], fill=(58, 36, 24, 255))                           # coffee
    d.rectangle([0.45 * U, 1.2 * U, 0.8 * U, 4.2 * U], fill=(250, 244, 228, 120))                         # highlight
    return _down(im, S)


def make_notebook():
    S, ppu = 2, 30
    tw, th = int(14 * ppu * S), int(10 * ppu * S)
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    U = ppu * S
    d.rounded_rectangle([0, 0, tw - 1, th - 1], int(0.9 * U), fill=(54, 62, 52, 255))
    d.rounded_rectangle([0.5 * U, 0.5 * U, tw - 0.5 * U, th - 0.5 * U], int(0.6 * U), outline=(70, 80, 66, 255), width=S)
    d.rectangle([11.2 * U, 0, 11.7 * U, th], fill=(190, 124, 58, 255))                                   # elastic band
    d.rectangle([1.4 * U, 1.3 * U, 6.4 * U, 3.7 * U], fill=(226, 216, 192, 255))                         # label
    for k in range(3):
        d.line([1.9 * U, (1.9 + k * 0.55) * U, 5.9 * U - k * 0.8 * U, (1.9 + k * 0.55) * U], fill=(120, 108, 90, 255), width=S)
    d.line([3.0 * U, 8.1 * U, 10.0 * U, 3.9 * U], fill=(28, 28, 32, 255), width=int(0.42 * U))            # pen
    d.line([9.1 * U, 4.45 * U, 10.0 * U, 3.9 * U], fill=(214, 140, 60, 255), width=int(0.42 * U))
    d.line([3.0 * U, 8.1 * U, 3.35 * U, 7.9 * U], fill=(170, 170, 180, 255), width=int(0.3 * U))
    return _down(im, S)


def make_controller():
    """Generic two-tone gamepad (white shell, black underbelly, bumpers and sticks), based on the shape language of
    modern pads: wide boomerang body, tall rounded grips, trapezoid touchpad with a blue light bar, parallel sticks,
    tone-on-tone face buttons. Standing prop, front view. 8 x 5.1 units, no logos."""
    S, ppu = 2, 100
    W, H = 8.0, 5.1
    w, h = int(W * ppu * S), int(H * ppu * S)
    U = ppu * S
    rgb = np.zeros((h, w, 3), np.float32)
    P = lambda pts: [(x * U, y * U) for x, y in pts]
    # black shoulder bumpers sit behind and above the body
    bumps = []
    for x0 in (0.5, 5.7):
        bm = _mask(w, h, lambda d, x0=x0: d.rounded_rectangle([x0 * U, 0.02 * U, (x0 + 1.8) * U, 0.78 * U], int(0.34 * U), fill=255))
        _paint(rgb, bm, (36, 36, 44), 0.2 * U, amb=0.5, gain=0.9, spec=40, rim=(22, 18, 14))
        bumps.append(bm)
    # body silhouette: wide centre with two tall grips that splay slightly
    def sil(d):
        d.rounded_rectangle([0.75 * U, 0.42 * U, 7.25 * U, 3.6 * U], int(1.3 * U), fill=255)
        d.polygon(P([(0.0, 1.7), (1.3, 1.1), (2.4, 2.6), (2.1, 4.2), (1.5, 5.0), (0.55, 5.0), (0.02, 3.9)]), fill=255)
        d.polygon(P([(8.0, 1.7), (6.7, 1.1), (5.6, 2.6), (5.9, 4.2), (6.5, 5.0), (7.45, 5.0), (7.98, 3.9)]), fill=255)
    body = _mask(w, h, sil)
    body = Image.fromarray(cv2.GaussianBlur(np.array(body), (0, 0), 0.22 * U)).point(lambda v: 255 if v > 127 else 0)
    body = Image.fromarray(cv2.GaussianBlur(np.array(body), (0, 0), 0.04 * U))
    _paint(rgb, body, (244, 246, 250), 0.55 * U, amb=0.38, gain=0.80, spec=48, k=4.0, rim=(24, 20, 16))
    # black underbelly crescent between the grips
    belly = _mask(w, h, lambda d: d.ellipse([1.9 * U, 3.05 * U, 6.1 * U, 4.5 * U], fill=255))
    bm_ = (np.array(belly, np.float32) / 255) * (np.array(body, np.float32) / 255)
    # touchpad (trapezoid) with a blue light bar hugging it
    tp = _mask(w, h, lambda d: d.polygon(P([(2.75, 0.78), (5.25, 0.78), (5.02, 1.85), (2.98, 1.85)]), fill=255))
    _paint(rgb, tp, (226, 230, 240), 0.10 * U, amb=0.65, gain=0.45, spec=20, rim=(10, 8, 6))
    glow = np.zeros((h, w), np.float32)
    lb = Image.new('L', (w, h), 0)
    dl = ImageDraw.Draw(lb)
    dl.line(P([(2.62, 0.7), (2.9, 1.95)]), fill=255, width=int(0.15 * U))
    dl.line(P([(5.38, 0.7), (5.1, 1.95)]), fill=255, width=int(0.15 * U))
    dl.line(P([(2.7, 0.7), (5.3, 0.7)]), fill=200, width=int(0.05 * U))
    lbm = np.array(lb, np.float32) / 255
    glow = cv2.GaussianBlur(lbm, (0, 0), 0.10 * U)
    rgb += glow[:, :, None] * np.array([40, 90, 190], np.float32)[None, None, :] * 1.6
    rgb = np.clip(rgb * (1 - lbm[:, :, None] * 0.9) + lbm[:, :, None] * np.array([130, 190, 255], np.float32) * 0.9, 0, 255)
    # small system buttons and mute/centre button
    for cx, cy, ww in ((2.28, 1.25, 0.28), (5.72, 1.25, 0.28)):
        pm = _mask(w, h, lambda d, cx=cx, cy=cy, ww=ww: d.rounded_rectangle([(cx - ww) * U, (cy - 0.08) * U, (cx + ww) * U, (cy + 0.08) * U], int(0.08 * U), fill=255))
        _paint(rgb, pm, (196, 200, 210), 0.05 * U, amb=0.6, gain=0.5)
    cm_ = _mask(w, h, lambda d: d.ellipse([3.75 * U, 1.95 * U, 4.25 * U, 2.23 * U], fill=255))
    _paint(rgb, cm_, (210, 214, 224), 0.07 * U, amb=0.6, gain=0.5, spec=30, rim=(10, 8, 6))
    ml = _mask(w, h, lambda d: d.ellipse([3.9 * U, 2.75 * U, 4.1 * U, 2.92 * U], fill=255))
    _paint(rgb, ml, (255, 150, 60), 0.04 * U, amb=0.9, gain=0.2)
    # parallel analog sticks: dark rubber caps in recessed wells
    for cx in (2.95, 5.05):
        well = _mask(w, h, lambda d, cx=cx: d.ellipse([(cx - 0.6) * U, 1.85 * U, (cx + 0.6) * U, 3.05 * U], fill=255))
        _paint(rgb, well, (20, 20, 26), 0.10 * U, amb=0.6, gain=0.4, rim=(0, 0, 0))
        cap = _mask(w, h, lambda d, cx=cx: d.ellipse([(cx - 0.46) * U, 1.97 * U, (cx + 0.46) * U, 2.93 * U], fill=255))
        _paint(rgb, cap, (44, 44, 54), 0.34 * U, amb=0.40, gain=0.95, spec=70, k=5.0)
        tx = np.array(Image.new('L', (w, h), 0))
        rg = Image.new('L', (w, h), 0)
        dr = ImageDraw.Draw(rg)
        for r_ in (0.16, 0.26, 0.36):
            dr.ellipse([(cx - r_) * U, 2.45 * U - r_ * U, (cx + r_) * U, 2.45 * U + r_ * U], outline=255, width=max(int(0.012 * U), 1))
        ra = np.array(rg, np.float32)[:, :, None] / 255
        rgb = rgb * (1 - 0.35 * ra) + 0.35 * ra * 12
    # d-pad (left) and four face buttons (right): white on white, separated by shadow rims and a faint glyph
    def pad_part(mk_img, col=(236, 238, 244), depth=0.07):
        m_ = np.array(mk_img, np.float32) / 255
        ring = cv2.GaussianBlur(m_, (0, 0), 0.05 * U)
        sh_ = np.clip(cv2.dilate(m_, np.ones((int(0.06 * U) | 1,) * 2)) - m_, 0, 1)
        nonlocal_rgb = rgb
        nonlocal_rgb *= (1 - 0.45 * cv2.GaussianBlur(sh_, (0, 0), 0.025 * U))[:, :, None]
        _paint(nonlocal_rgb, mk_img, col, depth * U, amb=0.55, gain=0.55, spec=26)
    dp = _mask(w, h, lambda d: (d.rounded_rectangle([1.06 * U, 1.0 * U, 2.14 * U, 1.38 * U], int(0.09 * U), fill=255),
                               d.rounded_rectangle([1.43 * U, 0.65 * U, 1.77 * U, 1.73 * U], int(0.09 * U), fill=255)))
    pad_part(dp)
    for bx, by in ((6.4, 0.84), (6.8, 1.2), (6.0, 1.2), (6.4, 1.56)):
        bm = _mask(w, h, lambda d, bx=bx, by=by: d.ellipse([(bx - 0.22) * U, (by - 0.22) * U, (bx + 0.22) * U, (by + 0.22) * U], fill=255))
        pad_part(bm)
    gl = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    dg = ImageDraw.Draw(gl, 'RGBA')
    g = (150, 156, 172, 200)
    wl = max(int(0.022 * U), 1)
    bx, by = 6.4, 0.84
    dg.polygon(P([(bx, by - 0.12), (bx + 0.11, by + 0.08), (bx - 0.11, by + 0.08)]), outline=g, width=wl)           # generic shapes
    dg.ellipse([(6.8 - 0.1) * U, (1.2 - 0.1) * U, (6.8 + 0.1) * U, (1.2 + 0.1) * U], outline=g, width=wl)
    dg.line(P([(6.4 - 0.1, 1.56 - 0.1), (6.4 + 0.1, 1.56 + 0.1)]), fill=g, width=wl)
    dg.line(P([(6.4 - 0.1, 1.56 + 0.1), (6.4 + 0.1, 1.56 - 0.1)]), fill=g, width=wl)
    dg.rectangle([(6.0 - 0.09) * U, (1.2 - 0.09) * U, (6.0 + 0.09) * U, (1.2 + 0.09) * U], outline=g, width=wl)
    out = np.array(gl)
    al = np.maximum.reduce([np.array(body)] + [np.array(m_) for m_ in bumps])
    rgba = np.dstack([out, cv2.GaussianBlur(al, (0, 0), 0.5 * S)])
    return cv2.resize(rgba, (int(W * ppu), int(H * ppu)), interpolation=cv2.INTER_AREA)


def make_window():
    """Back-wall window with the blinds raised: sunny blue sky with clouds, cream frame. 32 x 74 units."""
    S, ppu = 2, 20
    w, h = 32 * ppu * S, 74 * ppu * S
    U = ppu * S
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    sky = np.array([78, 142, 226], np.float32) * (1 - yy[:, :, None]) + np.array([182, 218, 250], np.float32) * yy[:, :, None]
    sky = np.broadcast_to(sky, (h, w, 3)).copy()
    sun = np.exp(-(((xx - 0.30) / 0.40) ** 2 + ((yy - 0.16) / 0.20) ** 2))
    sky += sun[:, :, None] * np.array([70, 62, 40], np.float32)
    rs = np.random.RandomState(41)
    clouds = np.zeros((h, w), np.float32)
    shade = np.zeros((h, w), np.float32)
    for _ in range(46):                                   # puffy cumulus: clusters of soft ellipses, denser lower down
        cy = (0.12 + 0.80 * rs.rand() ** 0.8) * h
        cx = rs.uniform(0.0, 1.0) * w
        for _k in range(rs.randint(3, 6)):
            rx, ry = rs.uniform(2.2, 5.5) * U, rs.uniform(1.1, 2.4) * U
            ox, oy = rs.uniform(-3.5, 3.5) * U, rs.uniform(-0.8, 0.8) * U
            cv2.ellipse(clouds, (int(cx + ox), int(cy + oy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.ellipse(shade, (int(cx + ox), int(cy + oy + 0.5 * ry)), (int(rx * 0.9), int(ry * 0.7)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    clouds = cv2.GaussianBlur(clouds, (0, 0), 0.55 * U)
    shade = cv2.GaussianBlur(shade, (0, 0), 0.7 * U)
    cl = np.clip(clouds * 1.15, 0, 1)
    ccol = np.array([252, 252, 255], np.float32)[None, None, :] * (1 - 0.18 * np.clip(shade - 0.2, 0, 1)[:, :, None])
    ccol = ccol * np.array([1.0, 1.0, 1.0], np.float32) + (1 - np.clip(shade, 0, 1))[:, :, None] * np.array([2, 2, 0], np.float32)
    sky = sky * (1 - cl[:, :, None]) + ccol * cl[:, :, None]
    im = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    fr = int(1.7 * U)
    # blinds pulled all the way up: a tight stack of slats under the head rail
    y = fr
    for k in range(5):
        d.rectangle([fr, y, w - fr, y + int(0.75 * U)], fill=(226, 210, 180))
        d.rectangle([fr, y + int(0.55 * U), w - fr, y + int(0.75 * U)], fill=(186, 168, 136))
        y += int(0.95 * U)
    d.rectangle([fr, fr - int(0.2 * U), w - fr, fr + int(0.35 * U)], fill=(206, 190, 160))
    cream, cream_d = (226, 214, 188), (186, 172, 146)
    d.rectangle([0, 0, w, fr], fill=cream)
    d.rectangle([0, h - fr, w, h], fill=cream)
    d.rectangle([0, 0, fr, h], fill=cream)
    d.rectangle([w - fr, 0, w, h], fill=cream)
    d.rectangle([w // 2 - int(0.55 * U), 0, w // 2 + int(0.55 * U), h], fill=cream)                       # mullion
    d.rectangle([0, h // 2 - int(0.55 * U), w, h // 2 + int(0.55 * U)], fill=cream)                        # transom
    d.rectangle([fr - 3, fr - 3, w - fr + 3, fr], fill=cream_d)
    d.rectangle([fr - 3, h - fr, w - fr + 3, h - fr + 3], fill=cream_d)
    a = np.array(im, np.float32) * 0.72   # emissive but not clipped, so clouds and frame survive the bloom
    return cv2.resize(np.dstack([np.clip(a, 0, 255).astype(np.uint8), np.full((h, w), 255, np.uint8)]), (32 * ppu, 74 * ppu), interpolation=cv2.INTER_AREA)


def _build_props():
    P = {}
    # --- wall shelf and what sits on it (depth-of-field blur matches the wall)
    P['board'] = make_shelf_board()
    P['plant'] = _lit(_blur_a(make_plant(), 1.1), wall_light(15, 31, 201, 181, 480, 600))
    P['lens'] = _lit(_blur_a(make_lens(), 1.1), wall_light(-30, -23.6, 190.5, 181, 192, 285))
    P['books'] = _lit(_blur_a(make_books(), 1.1), wall_light(-4, 7, 186, 181, 330, 156))
    P['shelf_shadow'], P['shelf_shadow_pad'] = _soft_shadow(72, 3, 20, 0.30, 1.6)
    P['window'] = make_window()
    P['sill'] = make_shelf_board()
    P['sill_shadow'], P['sill_shadow_pad'] = _soft_shadow(40, 2, 20, 0.30, 1.4)
    # --- desk set
    P['mat'] = make_mat()
    P['kb'] = make_kb60()
    P['mouse'] = make_mouse()
    P['console'] = make_console()
    P['drawers'] = make_drawers()
    P['kb_shadow'], P['kb_shadow_pad'] = _soft_shadow(30, 11.25, 20, 0.55, 1.2)
    P['mouse_shadow'], P['mouse_shadow_pad'] = _soft_shadow(4.4, 7, 30, 0.55, 0.9)
    P['con_shadow'], P['con_shadow_pad'] = _soft_shadow(12, 8, 20, 0.55, 3.0)
    # left side of the desk: camera, mug, notebook; right side: controller
    P['camera'] = _lit_desk(_blur_a(make_camera_body(), 0.5), CAM_X[0], CAM_X[1], CAM_Z - 2, CAM_Z + 2)
    P['mug'] = _lit_desk(make_mug(), MUG_X[0], MUG_X[1], MUG_Z - 1, MUG_Z + 1)
    P['notebook'] = _lit_desk(make_notebook(), NB_X[0], NB_X[1], NB_Z[0], NB_Z[1])
    P['controller'] = _lit_desk(make_controller(), CTL_X[0], CTL_X[1], CTL_Z[0], CTL_Z[1])
    P['cam_shadow'], P['cam_shadow_pad'] = _soft_shadow(19, 5, 20, 0.55, 1.2)
    P['mug_shadow'], P['mug_shadow_pad'] = _soft_shadow(8.0, 3.8, 30, 0.55, 1.1)
    P['nb_shadow'], P['nb_shadow_pad'] = _soft_shadow(21, 15, 20, 0.5, 0.9)
    P['ctl_shadow'], P['ctl_shadow_pad'] = _soft_shadow(7.2, 4.8, 30, 0.5, 0.7)
    return P


_PROPS = {}
KB_X, KB_Z0, KB_Z1 = (-23.0, 7.0), 57.0, 68.25
MOUSE_X, MOUSE_Z = (17.8, 22.2), (61.5, 68.5)
CON_X, CON_ZF, CON_ZB = (31.0, 43.0), 33.0, 26.0       # console: wide white face toward the camera, right of the monitor
CTL_X, CTL_Z, CTL_ANG = (32.5, 40.5), (54.0, 59.0), 14.0   # controller on the desk in front of the console
CAM_X, CAM_Z, CAM_H = (-67.0, -48.0), 36.0, 10.5   # compact mirrorless camera (generic, no brand marks)                     # camera + telephoto lens standing on the desk, left of the monitor
MUG_X, MUG_Z, MUG_H = (-44.0, -35.9), 44.0, 6.9
NB_X, NB_Z, NB_ANG = (-60.0, -39.0), (54.0, 69.0), -9.0  # notebook + pen, left of the mouse pad
WIN_X, WIN_Y = (-76.0, -50.0), (118.0, 180.0)           # window on the back wall (only visible in the wide opening shot)


def _flat(x0, x1, z0, z1, y):
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)


def _rot_flat(xr, zr, ang_deg, y):
    """A flat quad on the desk (TL, TR, BR, BL), rotated about its centre by ang_deg."""
    cx, cz = (xr[0] + xr[1]) / 2, (zr[0] + zr[1]) / 2
    a = math.radians(ang_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = np.array([[xr[0] - cx, zr[0] - cz], [xr[1] - cx, zr[0] - cz], [xr[1] - cx, zr[1] - cz], [xr[0] - cx, zr[1] - cz]]) @ R.T
    return np.column_stack([pts[:, 0] + cx, np.full(4, y), pts[:, 1] + cz])


def _vert(x0, x1, ytop, ybot, z):
    return np.array([[x0, ytop, z], [x1, ytop, z], [x1, ybot, z], [x0, ybot, z]], float)


def draw_room_back(cv, cam):
    """Wall shelf with props (drawn right after the wall)."""
    if not _PROPS:
        _PROPS.update(_build_props())
    Pp = _PROPS
    pad = Pp['shelf_shadow_pad']
    C.quad(cv, cam, Pp['shelf_shadow'], _vert(-36 - pad, 36 + pad, 178.4 + pad - 1.0, 178.4 - 3 - pad - 1.0, 0.15), alpha=True)
    # window with blinds (emissive), sill and a soft shadow under the sill
    C.quad(cv, cam, Pp['window'], _vert(WIN_X[0], WIN_X[1], WIN_Y[1], WIN_Y[0], 0.2), alpha=True)
    sp = Pp['sill_shadow_pad']
    C.quad(cv, cam, Pp['sill_shadow'], _vert(WIN_X[0] - 4 - sp, WIN_X[1] + 4 + sp, WIN_Y[0] - 0.4 + sp, WIN_Y[0] - 2.4 - sp - 1.0, 0.15), alpha=True)
    C.quad(cv, cam, Pp['sill'], _vert(WIN_X[0] - 2, WIN_X[1] + 2, WIN_Y[0], WIN_Y[0] - 1.6, 2.0), alpha=True)
    C.quad(cv, cam, Pp['lens'], _vert(-30, -23.6, 190.5, 181, 3.0), alpha=True)
    C.quad(cv, cam, Pp['books'], _vert(-4, 7, 186.2, 181, 3.0), alpha=True)
    C.quad(cv, cam, Pp['plant'], _vert(15, 31, 201, 181, 3.0), alpha=True)
    C.quad(cv, cam, Pp['board'], _vert(-36, 36, 181, 178.4, 6.0), alpha=True)


def draw_desk_set(cv, cam):
    """Desk front edge, console, mat (drawn after the desk, before the monitor)."""
    Pp = _PROPS
    # desk front edge + dark under-desk so it does not float
    C.quad(cv, cam, np.full((8, 8, 4), (112, 72, 44, 255), np.uint8), _vert(-120, 120, 75, 69.5, 85), alpha=True, cull=False)
    under = np.zeros((64, 8, 4), np.uint8)
    under[:, :, :3] = (np.linspace(10, 74, 64)[:, None, None] ** 1.0 * np.array([1.0, 0.62, 0.40])[None, None, :]).astype(np.uint8)
    under[:, :, 3] = 255
    C.quad(cv, cam, under, _vert(-120, 120, 69.5, 0, 85), alpha=True, cull=False)
    C.quad(cv, cam, Pp['drawers'], _vert(22, 112, 69.0, 12, 84.6), alpha=True, cull=False)
    # console contact shadow + console (front face + the side that turns toward the camera)
    pad = Pp['con_shadow_pad']
    C.quad(cv, cam, Pp['con_shadow'], _flat(CON_X[0] - pad + 1.5, CON_X[1] + pad + 1.5, CON_ZB - pad, CON_ZF + pad, 75.05), alpha=True, cull=False)
    # thickness: a darker copy of the same silhouette further back (parallax makes it read as a solid tower)
    back = Pp['console'].copy()
    back[:, :, :3] = (back[:, :, :3].astype(np.float32) * 0.66).astype(np.uint8)
    C.quad(cv, cam, back, _vert(CON_X[0] + 0.5, CON_X[1] + 0.5, 109.6, 75.1, CON_ZB), alpha=True, cull=False)
    C.quad(cv, cam, Pp['console'], _vert(CON_X[0], CON_X[1], 110.1, 75.1, CON_ZF), alpha=True, cull=False)
    # mat
    C.quad(cv, cam, Pp['mat'], _flat(-31, 31, 51, 82, 75.25), alpha=True, cull=False)
    # left side of the desk: notebook with a pen, camera with telephoto lens, mug
    nb = _rot_flat(NB_X, NB_Z, NB_ANG, 75.3)
    pad = Pp['nb_shadow_pad']
    C.quad(cv, cam, Pp['nb_shadow'], _rot_flat((NB_X[0] - pad + 0.4, NB_X[1] + pad + 0.4), (NB_Z[0] - pad + 0.6, NB_Z[1] + pad + 0.6), NB_ANG, 75.28), alpha=True, cull=False)
    C.quad(cv, cam, Pp['notebook'], nb, alpha=True, cull=False)
    pad = Pp['cam_shadow_pad']
    C.quad(cv, cam, Pp['cam_shadow'], _flat(CAM_X[0] - pad + 1.2, CAM_X[1] + pad + 1.2, CAM_Z - 2.5 - pad, CAM_Z + 2.5 + pad, 75.05), alpha=True, cull=False)
    C.quad(cv, cam, Pp['camera'], _vert(CAM_X[0], CAM_X[1], 75.1 + CAM_H, 75.1, CAM_Z), alpha=True, cull=False)
    pad = Pp['mug_shadow_pad']
    C.quad(cv, cam, Pp['mug_shadow'], _flat(MUG_X[0] - pad + 0.9, MUG_X[1] + pad + 0.9, MUG_Z - 1.9 - pad, MUG_Z + 1.9 + pad, 75.05), alpha=True, cull=False)
    C.quad(cv, cam, Pp['mug'], _vert(MUG_X[0], MUG_X[1], 75.1 + MUG_H, 75.1, MUG_Z), alpha=True, cull=False)
    # right side: controller
    pad = Pp['ctl_shadow_pad']
    C.quad(cv, cam, Pp['ctl_shadow'], _rot_flat((CTL_X[0] - pad + 0.3, CTL_X[1] + pad + 0.3), (CTL_Z[0] - pad + 0.5, CTL_Z[1] + pad + 0.5), CTL_ANG, 75.28), alpha=True, cull=False)
    C.quad(cv, cam, Pp['controller'], _vert(CTL_X[0], CTL_X[1], 75.1 + 5.0, 75.1, 56.5), alpha=True, cull=False)


def draw_desk_front(cv, cam):
    """Keyboard + mouse with contact shadows (drawn after the monitor)."""
    Pp = _PROPS
    pad = Pp['kb_shadow_pad']
    C.quad(cv, cam, Pp['kb_shadow'], _flat(KB_X[0] - pad + 0.5, KB_X[1] + pad + 0.5, KB_Z0 - pad + 0.9, KB_Z1 + pad + 0.9, 75.3), alpha=True, cull=False)
    C.quad(cv, cam, Pp['kb'], _flat(KB_X[0], KB_X[1], KB_Z0, KB_Z1, 75.55), alpha=True, cull=False)
    pad = Pp['mouse_shadow_pad']
    C.quad(cv, cam, Pp['mouse_shadow'], _flat(MOUSE_X[0] - pad + 0.3, MOUSE_X[1] + pad + 0.3, MOUSE_Z[0] - pad + 0.5, MOUSE_Z[1] + pad + 0.5, 75.3), alpha=True, cull=False)
    C.quad(cv, cam, Pp['mouse'], _flat(MOUSE_X[0], MOUSE_X[1], MOUSE_Z[0], MOUSE_Z[1], 75.7), alpha=True, cull=False)


def draw_monitor_body_hq(cv, cam):
    """Rounded, tapered CRT body built from stacked rounded-rectangle slices, plus a stand."""
    foot = np.zeros((32, 64, 4), np.uint8)
    foot[:, :, :3] = (150, 137, 114)
    foot[:, :, 3] = 255
    foot[:, :, :3] = (foot[:, :, :3] * np.linspace(1.05, 0.62, 32)[:, None, None]).astype(np.uint8)
    C.quad(cv, cam, foot, np.array([[-17, 82.5, 44], [17, 82.5, 44], [20, 75.2, 47], [-20, 75.2, 47]], float), alpha=True, cull=False)
    n = 40
    for i in range(n + 1):
        u = i / n
        z = 12 + 38 * u
        sc = 0.74 + 0.26 * math.sin(u * math.pi / 2)         # taper toward the back, softly domed
        hw, hh = 24 * sc, 22 * (0.80 + 0.20 * math.sin(u * math.pi / 2))
        r = (5.5 + 9.0 * (1 - u)) * 1.0
        r = min(r, hw - 1, hh - 1)
        cx, cy = 0.0, 105.0
        pts = []
        for (ccx, ccy, a0) in ((hw - r, hh - r, 0), (-(hw - r), hh - r, 90), (-(hw - r), -(hh - r), 180), (hw - r, -(hh - r), 270)):
            for a_ in np.linspace(a0, a0 + 90, 7):
                pts.append([cx + ccx + r * math.cos(math.radians(a_)), cy + ccy + r * math.sin(math.radians(a_)), z])
        P = C._mscale(np.array(pts))
        px, zc = cam.proj(P)
        if (zc < 3).any():
            continue
        t = u ** 1.4
        col = tuple(int(c) for c in (np.array([66, 59, 51]) * (1 - t) + np.array([160, 147, 124]) * t))
        cv2.fillPoly(cv, [np.round(px * 16).astype(np.int32)], col, cv2.LINE_AA, 4)


