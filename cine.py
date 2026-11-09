"""Cinematic remake of Day 2 (shutter speed): a small lit 3D room rendered in software.
Pinhole camera, textured planes via homographies, baked window-blind light, CRT screen content drawn per frame,
bloom / grade / vignette / grain in post. Everything is computed per frame."""
import math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SSF = 1.5   # 3D supersampling factor
SS = 1.5    # CRT screen texture scale (1.0 = 800x600)
WS, HS = int(W * SSF), int(H * SSF)
FL = 1300.0
_MONO_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf',
    '/Library/Fonts/DejaVuSansMono-Bold.ttf',
    'C:/Windows/Fonts/consolab.ttf',
]
MONO = next((p for p in _MONO_CANDIDATES if os.path.exists(p)), os.environ.get('MONO_FONT', _MONO_CANDIDATES[0]))
_fc = {}


def mono(sz):
    if sz not in _fc:
        _fc[sz] = ImageFont.truetype(MONO, sz)
    return _fc[sz]


class SD:
    """ImageDraw wrapper: draw in 800x600 coordinates onto a SS-scaled image."""
    def __init__(self, im):
        self.d = ImageDraw.Draw(im)

    @staticmethod
    def _xy(xy):
        return (np.array(xy, float).ravel() * SS).tolist()

    @staticmethod
    def _w(w):
        return max(1, int(round(w * SS)))

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.d.rectangle(self._xy(xy), fill=fill, outline=outline, width=self._w(width))

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.d.ellipse(self._xy(xy), fill=fill, outline=outline, width=self._w(width))

    def line(self, xy, fill=None, width=1):
        self.d.line(self._xy(xy), fill=fill, width=self._w(width))

    def text(self, xy, s, font=None, fill=None, anchor=None):
        self.d.text(self._xy(xy), s, font=font, fill=fill, anchor=anchor)


def pst(im, arr, pos):
    im.paste(Image.fromarray(np.ascontiguousarray(arr)), (int(pos[0] * SS), int(pos[1] * SS)))


HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_PATH = os.environ.get('PHOTO', os.path.join(HERE, 'assets', 'photo.jpg'))
BRAND = 'Jorgeb_photography'                      # printed on the monitor's top bezel in every scene
TITLE = os.environ.get('CINE_TITLE', 'SHUTTER SPEED')   # video title, shown on the monitor bezel (bottom)


class _Scene:
    def __init__(self, dur):
        self.dur = dur


# scene durations in seconds (problem, menu, bad result, fix, dark/ISO, trick, CTA).
# When voice clips exist, replace these with clip length + ~0.3 s tail.
SCENE_DURS = [3.91, 5.61, 5.61, 3.2, 9.01, 4.59, 5.27]
SC = [_Scene(d) for d in SCENE_DURS]
OFF = 3.6   # intro shot before the original scenes
STARTS = OFF + np.cumsum([0] + [s.dur for s in SC])
TOTAL = float(os.environ.get('CINE_END', STARTS[-1]))
NFR = int(round(TOTAL * FPS))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, t0, d):
    return clamp((t - t0) / d)


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ camera
KEYS = [
    (0.0, (-6, 113, 119), (-6, 112, 45)),
    (OFF, (5, 114, 126), (0, 113, 45)),
    (OFF + 3.91, (-3, 113, 118), (0, 112, 45)),
    (OFF + 9.52, (2, 112, 112), (0, 112, 46)),
    (OFF + 15.13, (-2, 112, 110), (0, 111, 46)),
    (OFF + 18.33, (2, 112, 112), (0, 112, 46)),
    (OFF + 27.34, (-3, 113, 118), (0, 113, 45)),
    (OFF + 31.93, (0, 114, 122), (0, 113, 45)),
    (OFF + 37.2, (0, 115, 134), (0, 114, 45)),
]
_kt = np.array([k[0] for k in KEYS])
_kv = np.array([list(k[1]) + list(k[2]) for k in KEYS], dtype=float)
_km = _kv[1:].mean(0)
_kv[1:] = _km + 0.6 * (_kv[1:] - _km)


def cam_path(t):
    t = clamp(t, 0, _kt[-1] - 1e-6)
    i = int(np.searchsorted(_kt, t, 'right') - 1)
    i = min(i, len(_kt) - 2)
    dt = _kt[i + 1] - _kt[i]
    s = (t - _kt[i]) / dt

    def tang(j):
        a, b = max(j - 1, 0), min(j + 1, len(_kt) - 1)
        return (_kv[b] - _kv[a]) / (_kt[b] - _kt[a]) * dt * 0.8

    p0, p1, m0, m1 = _kv[i], _kv[i + 1], tang(i), tang(i + 1)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    v = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return v[:3], v[3:]


class Cam:
    def __init__(self, t):
        p, g = cam_path(t)
        p = p + np.array([0.6 * (0.14 * math.sin(t * 1.3) + 0.06 * math.sin(t * 2.9 + 1)),
                          0.6 * (0.10 * math.sin(t * 1.1 + 2) + 0.04 * math.sin(t * 3.3)), 0.0])
        g = g + np.array([0.6 * 0.09 * math.sin(t * 0.9 + 3), 0.6 * 0.08 * math.sin(t * 1.7), 0.0])
        f = g - p
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 1, 0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        a = 0.0018 * math.sin(t * 0.8) + 0.0007 * math.sin(t * 2.3)
        self.r, self.u = r * math.cos(a) + u * math.sin(a), -r * math.sin(a) + u * math.cos(a)
        self.f, self.p = f, p

    def proj(self, P):
        d = np.asarray(P, dtype=float) - self.p
        zc = d @ self.f
        zc_ = np.maximum(zc, 1e-3)
        x = SSF * (W / 2 + FL * (d @ self.r) / zc_)
        y = SSF * (H / 2 - FL * (d @ self.u) / zc_)
        return np.stack([x, y], 1), zc


def quad(cv, cam, tex, P, alpha=False, cull=True):
    """Warp a texture (TL,TR,BR,BL order) onto a world quad. cv: uint8 RGB canvas."""
    pts, zc = cam.proj(P)
    if (zc < 3).any():
        return
    if cull:
        a = 0.0
        for i in range(4):
            j = (i + 1) % 4
            a += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        if a <= 0:
            return
    x0, y0 = int(max(0, math.floor(pts[:, 0].min()))), int(max(0, math.floor(pts[:, 1].min())))
    x1, y1 = int(min(WS, math.ceil(pts[:, 0].max()))), int(min(HS, math.ceil(pts[:, 1].max())))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, (pts - [x0, y0]).astype(np.float32))
    w, h = x1 - x0, y1 - y0
    if not alpha:
        sub = np.ascontiguousarray(cv[y0:y1, x0:x1])
        cv2.warpPerspective(tex, M, (w, h), dst=sub, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        cv[y0:y1, x0:x1] = sub
    else:
        wr = cv2.warpPerspective(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = wr[:, :, 3:4].astype(np.float32) / 255.0
        sub = cv[y0:y1, x0:x1].astype(np.float32)
        cv[y0:y1, x0:x1] = np.clip(sub * (1 - a) + wr[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)


def poly(cv, cam, P, color, center_normal):
    pts, zc = cam.proj(P)
    if (zc < 3).any():
        return
    c = np.mean(P, 0)
    if np.dot(center_normal, cam.p - c) <= 0:
        return
    cv2.fillConvexPoly(cv, np.round(pts * 16).astype(np.int32), color, cv2.LINE_AA, 4)


# ------------------------------------------------------------------ textures (baked once)
def blind_light(U, V, period, ang, phase=0.0, soft=0.07):
    a = math.radians(ang)
    u = (U * math.cos(a) + V * math.sin(a)) / period + phase
    ph = u % 1.0
    return sstep(0.04, 0.04 + soft, ph) * (1 - sstep(0.46, 0.46 + soft + 0.03, ph))


def make_wall():
    px = 6
    x = np.linspace(-110, 110, 220 * px)[None, :]
    y = np.linspace(260, 20, 240 * px)[:, None]
    X, Y = np.broadcast_arrays(x, y)
    rs = np.random.RandomState(1)
    noise = cv2.resize(rs.rand(26, 24).astype(np.float32), (X.shape[1], X.shape[0]), interpolation=cv2.INTER_CUBIC)
    base = np.array([176, 122, 88], np.float32)[None, None, :] * (0.93 + 0.14 * noise)[:, :, None]
    base *= (1 - 0.55 * np.clip((Y - 80) / 180, 0, 1))[:, :, None]
    band = blind_light(X, Y, 34.0, -33, 0.15, 0.24)
    env = 0.40 + 0.60 * sstep(-0.6, 0.9, (-(X + 25) * 0.012 + (Y - 120) * 0.0 + 0.55 - np.abs(Y - 130) * 0.004))
    light = 0.70 + 0.28 * band * env
    # leaf shadows (right side) only matter where there is light
    sh = np.zeros(X.shape, np.float32)
    for k in range(26):
        cx, cy = rs.uniform(48, 98), rs.uniform(95, 205)
        ang = rs.uniform(-1.1, 1.1)
        axes = (int(rs.uniform(9, 17) * px), int(rs.uniform(3, 5.5) * px))
        c = (int((cx + 110) * px), int((260 - cy) * px))
        cv2.ellipse(sh, c, axes, math.degrees(ang), 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.line(sh, (int(158 * px), int((260 - 70) * px)), (int(190 * px), int((260 - 200) * px)), 1.0, 6)
    sh = cv2.GaussianBlur(sh, (0, 0), 5)
    light *= (1 - 0.82 * sh)
    # soft monitor shadow behind it
    ms = np.exp(-(((X - 6) / 34) ** 4 + ((Y - 104) / 30) ** 4))
    light *= (1 - 0.55 * ms)
    warm = np.stack([np.ones_like(light), 0.80 + 0.2 * (1 - light), 0.55 + 0.4 * (1 - light)], 2)
    tex = base * light[:, :, None] * (0.9 + 0.1 * warm) * 1.18
    tex = cv2.GaussianBlur(tex, (0, 0), 2.2)  # depth of field
    return np.clip(tex, 0, 255).astype(np.uint8)


def make_desk():
    px = 4
    x = np.linspace(-120, 120, 240 * px)[None, :]
    z = np.linspace(0, 85, 85 * px)[:, None]
    X, Z = np.broadcast_arrays(x, z)
    rs = np.random.RandomState(2)
    g = rs.randn(Z.shape[0], X.shape[1]).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=50, sigmaY=0.8)
    g = g / (np.abs(g).max() + 1e-6)
    base = np.array([118, 74, 44], np.float32)[None, None, :] * (1 + 0.28 * g)[:, :, None]
    band = blind_light(X, Z, 30.0, 18, 0.35, 0.24)
    env = 0.45 + 0.55 * sstep(-80, 40, 40 - X)
    light = 0.68 + 0.28 * band * env
    ms = np.exp(-(((X - 8) / 30) ** 4 + ((Z - 32) / 26) ** 4))
    light *= (1 - 0.65 * ms)
    ks = np.exp(-(((X - 4) / 26) ** 4 + ((Z - 68) / 14) ** 4))
    light *= (1 - 0.45 * ks)
    tex = base * light[:, :, None] * 1.15
    tex = cv2.GaussianBlur(tex, (0, 0), 1.0)
    return np.clip(tex, 0, 255).astype(np.uint8)


def make_bezel():
    pxcm = 10
    w, h = 48 * pxcm, 44 * pxcm
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([168, 154, 130], np.float32)[None, None, :] * (0.90 + 0.10 * (1 - yy / h))[:, :, None]
    band = blind_light(xx * 0.1, yy * 0.1, 17.0, -33, 0.0, 0.1)
    base *= (0.93 + 0.10 * band)[:, :, None]
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, 'RGBA')
    d.rounded_rectangle([40, 55, 440, 372], radius=34, fill=(26, 22, 20), outline=(120, 108, 92), width=3)
    d.rounded_rectangle([46, 61, 434, 366], radius=30, outline=(0, 0, 0, 120), width=5)
    d.ellipse([382, 390, 398, 406], fill=(120, 255, 120))
    for k in range(6):
        d.line([60 + k * 18, 396, 60 + k * 18, 414], fill=(120, 108, 92), width=3)
    ttl, fs = TITLE.upper(), 15
    while fs > 9 and mono(fs).getlength(ttl) > 190:   # keep it between the vent slots and the power light
        fs -= 1
    d.text((240, 400), ttl, font=mono(fs), fill=(104, 94, 80), anchor='mm')
    d.text((240, 27), BRAND, font=mono(15), fill=(104, 94, 80), anchor='mm')   # same font/colour as the bottom label
    return np.array(im)


def make_keyboard():
    w, h = 400, 220
    im = Image.new('RGB', (w, h), (190, 176, 150))
    d = ImageDraw.Draw(im)
    for r in range(5):
        n = 14 if r < 4 else 1
        for c in range(n):
            if r == 4:
                d.rounded_rectangle([90, 22 + r * 40, 310, 54 + r * 40], 5, fill=(214, 202, 176))
                break
            x0 = 10 + c * 27.5 + (r % 2) * 5
            d.rounded_rectangle([x0, 18 + r * 40, x0 + 24, 50 + r * 40], 5, fill=(214, 202, 176))
    a = np.array(im, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = blind_light(xx * 0.1, yy * 0.1, 15.0, 18, 0.2, 0.1)
    a *= (0.90 + 0.12 * band)[:, :, None]
    return np.clip(a, 0, 255).astype(np.uint8)


WALL, DESK, BEZEL, KEYB = make_wall(), make_desk(), make_bezel(), make_keyboard()


# ------------------------------------------------------------------ wall headline (typewriter)
WLINES = [
    [("FOOTBALL PHOTOS?", 0.25), ("BLURRY.", 1.1), ("ONE NUMBER.", 2.68)],
    [("SHUTTER SPEED.", 0.25), ("FOOTBALL IS FAST.", 1.5), ("MIN: 1/1000", 3.7)],
    [("SHOOTING AT 1/250:", 0.3), ("SHARP STADIUM.", 3.0), ("SMEARED PLAYER.", 4.38)],
    [("SET 1/1000.", 0.25), ("SHARP.", 1.5)],
    [("NOW IT'S DARK?", 0.25), ("GOOD.", 1.6), ("OPEN THE LENS.", 3.7), ("RAISE ISO.", 5.74)],
    [("GRAIN: FIXABLE.", 0.3), ("BLUR: NEVER.", 1.66), ("THAT'S THE TRICK.", 2.68)],
    [("TRY IT FRIDAY NIGHT.", 0.3), ("COMMENT SETTINGS", 1.66), ("SETUP IN YOUR DMs", 2.68)],
]
CPS = 17.0
TXT_COL = (255, 96, 52)


def typed(lines, t):
    out = []
    for s, t0 in lines:
        n = int(clamp((t - t0) * CPS, 0, len(s)))
        out.append((s[:n], n < len(s) and t >= t0, t >= t0))
    return out


def wall_text(si, tl, dur):
    th, tw = 330, 1000
    sz, lh = 64, 76
    lines = typed(WLINES[si], tl)
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y0 = 14
    fin = max(WLINES[si], key=lambda z: len(z[0]))[0]
    xoff = int((1000 - mono(sz).getlength(fin)) / 2)
    last = None
    for k, (s, typing, started) in enumerate(lines):
        if started:
            d.text((xoff, y0 + k * lh), s, font=mono(sz), fill=TXT_COL + (255,))
            last = (k, s)
    if last is not None:
        k, s = last
        tx = xoff + mono(sz).getlength(s)
        if int(tl * 2.6) % 2 == 0 or lines[k][1]:
            d.rectangle([tx + 3, y0 + k * lh + 6, tx + 30, y0 + k * lh + sz + 2], fill=TXT_COL + (255,))
    a = np.array(im, np.float32)
    al = a[:, :, 3]
    glow = cv2.GaussianBlur(al, (0, 0), 9)
    al2 = np.maximum(al, glow * 0.8)
    out_fade = 1 - smooth((tl - (dur - 0.3)) / 0.3)
    flick = 0.93 + 0.07 * math.sin(tl * 37) * math.sin(tl * 11)
    al2 = al2 * out_fade * flick
    rgb = np.empty((th, tw, 3), np.float32)
    rgb[:] = TXT_COL
    rgb[al > 200] = (255, 150, 110)
    rgba = np.dstack([rgb, al2]).astype(np.uint8)
    return rgba


# ------------------------------------------------------------------ photo (what is on the CRT)
PWD, PHT = 720, 400


# ------------------------------------------------------------------ the user's real photo
_RP = {}
_BASE_W, _BASE_H = 1440, 960


def _rp_base():
    if 'sharp' not in _RP:
        im = cv2.cvtColor(cv2.imread(PHOTO_PATH), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (_BASE_W, _BASE_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        _RP['sharp'] = im
        k = 1.0
        _RP['blur_all'] = cv2.filter2D(cv2.copyMakeBorder(im, 0, 0, 90, 90, cv2.BORDER_REFLECT), -1,
                                       np.ones((1, 76), np.float32) / 76)[:, 90:-90]
        m = np.zeros((_BASE_H, _BASE_W), np.float32)
        cv2.rectangle(m, (330, 330), (980, 760), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 38)
        bp = cv2.filter2D(cv2.copyMakeBorder(im, 0, 0, 160, 160, cv2.BORDER_REFLECT), -1,
                          np.ones((1, 130), np.float32) / 130)[:, 160:-160]
        _RP['blur_players'] = im * (1 - m[:, :, None]) + bp * m[:, :, None]
    return _RP


def rphoto(mode, tl, dur, bright=1.0, grain=0.0, seed=0):
    """Return (720x400 uint8 crop with slow push-in, mapper from base coords -> photo px)."""
    R = _rp_base()
    src = R[mode]
    z = 1.0 + 0.045 * clamp(tl / max(dur, 0.1))
    cw, ch = _BASE_W / z, 800.0 / z
    x0 = (_BASE_W - cw) / 2 + 0.0
    y0 = 80 + (800 - ch) / 2
    OW, OH = int(PWD * SS), int(PHT * SS)
    M = np.float32([[OW / cw, 0, -x0 * OW / cw], [0, OH / ch, -y0 * OH / ch]])
    out = cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_AREA if z < 1.01 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = out * bright
    if grain > 0:
        rs = np.random.RandomState(seed)
        out = out + rs.randn(OH, OW, 1).astype(np.float32) * grain + rs.randn(OH, OW, 3).astype(np.float32) * grain * 0.3
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    mapper = lambda bx, by: ((bx - x0) * PWD / cw, (by - y0) * PHT / ch)
    return out, mapper


# ------------------------------------------------------------------ CRT screen content (800x600)
AMB, DIMA, REDS, GRN = (255, 176, 66), (150, 96, 40), (255, 92, 64), (150, 255, 120)
BGS = (20, 11, 7)


def sd(d, xy, s, sz=22, fill=AMB, anchor='lm'):
    d.text(xy, s, font=mono(int(round(sz * SS))), fill=fill, anchor=anchor)


def screen_base(fr):
    im = Image.new('RGB', (int(800 * SS), int(600 * SS)), BGS)
    d = SD(im)
    d.rectangle([0, 0, 800, 36], fill=(44, 24, 10))
    sd(d, (14, 18), "SHUTTER_PRO v2.1", 22)
    sd(d, (786, 18), f"FRM {fr:04d}", 20, DIMA, 'rm')
    return im, d


def readouts(d, sh, ap, iso, hl=None, hcol=AMB):
    cells = [("SHUTTER", sh), ("APERTURE", ap), ("ISO", iso)]
    for i, (k, v) in enumerate(cells):
        x0 = 40 + i * 245
        on = (hl == i)
        d.rectangle([x0, 496, x0 + 225, 572], fill=hcol if on else (36, 20, 9), outline=DIMA if not on else hcol, width=2)
        sd(d, (x0 + 12, 515), k, 20, BGS if on else DIMA)
        sd(d, (x0 + 12, 548), v, 34, BGS if on else AMB)


def paste_photo(im, arr, pos=(40, 50)):
    pst(im, arr, pos)
    d = SD(im)
    d.rectangle([pos[0] - 2, pos[1] - 2, pos[0] + arr.shape[1] / SS + 1, pos[1] + arr.shape[0] / SS + 1], outline=DIMA, width=2)
    return d


def brackets(d, box, col, w=4, L=26):
    x0, y0, x1, y1 = box
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([cx, cy, cx + sx * L, cy], fill=col, width=w)
        d.line([cx, cy, cx, cy + sy * L], fill=col, width=w)


def label(d, xy, text, col, sz=24, p=1.0):
    if p <= 0:
        return
    w = mono(sz).getlength(text) + 22
    x, y = xy
    d.rectangle([x, y - sz * 0.8, x + w, y + sz * 0.8], fill=(14, 8, 5), outline=col, width=3)
    sd(d, (x + 11, y), text, sz, col)


def screen_scene(si, tl, dur, fr, gt):
    if si == -1:
        im = intro_screen(tl, fr)
        d = SD(im)
    else:
        im, d = screen_base(fr)
    ph = tl * 9.0
    if si == -1:
        pass
    elif si == 0:
        arr, _ = rphoto('blur_all', tl, dur)
        d = paste_photo(im, arr)
        sd(d, (52, 66), "IMG_0412.CR3", 18, (255, 220, 170))
        readouts(d, "1/125", "f/2.8", "3200", 0 if tl > 0.4 else None, REDS)
        if tl > 0.6:
            label(d, (440, 112), "x BLURRY", REDS, 28, 1)
        if tl > 2.68:
            if int(tl * 3) % 2 == 0:
                sd(d, (400, 300), "ONE NUMBER", 34, (255, 230, 190), 'mm')
    elif si == 1:
        sd(d, (40, 80), "> SELECT SHUTTER SPEED", 26)
        speeds = ["1/125", "1/250", "1/500", "1/1000", "1/2000"]
        tsel = 3.7
        idx = int(clamp((tl - 2.0) / 1.7 * 3.99, 0, 3)) if tl >= 2.0 else 0
        for i, s in enumerate(speeds):
            y = 160 + i * 66
            on = (i == idx) and tl >= 1.0
            done = tl >= tsel
            col = AMB
            if done and i == 3:
                d.rectangle([40, y - 28, 480, y + 28], fill=AMB)
                sd(d, (60, y), "> " + s, 36, BGS)
                sd(d, (500, y), "< FOOTBALL MIN", 22, GRN)
            else:
                if on and not done:
                    d.rectangle([40, y - 28, 480, y + 28], outline=AMB, width=3)
                    sd(d, (60, y), "> " + s, 36, AMB)
                else:
                    sd(d, (60, y), "  " + s, 36, DIMA if not done else (110, 70, 30))
                if done and i < 3:
                    sd(d, (500, y), "TOO SLOW", 20, REDS)
                elif done and i == 4:
                    sd(d, (500, y), "FAST", 20, GRN)
        readouts(d, speeds[idx] if tl < tsel else "1/1000", "f/2.8", "3200", 0, AMB)
    elif si == 2:
        arr, mp = rphoto('blur_players', tl, dur)
        d = paste_photo(im, arr)
        readouts(d, "1/250", "f/2.8", "3200", 0, REDS)
        if tl > 3.0:
            label(d, (56, 90), "SHARP STADIUM ok", GRN, 22, 1)
            brackets(d, (52, 66, 480, 210), GRN, 3, 18)
        if tl > 4.38:
            gx, gy = mp(650, 740)
            label(d, (clamp(40 + gx - 130, 60, 520), 50 + gy + 28), "x SMEARED PLAYER", REDS, 22, 1)
    elif si == 3:
        tf = 1.15
        arr, mp = rphoto('blur_players' if tl < tf else 'sharp', tl, dur)
        d = paste_photo(im, arr)
        readouts(d, "1/1000", "f/2.8", "3200", 0, GRN)
        if tl >= tf:
            bx0, by0 = mp(690, 360)
            bx1, by1 = mp(870, 690)
            brackets(d, (40 + bx0, 50 + by0, 40 + bx1, 50 + by1), GRN, 4, 28)
            label(d, (56, 90), "ok SHARP", GRN, 30, 1)
        if tf <= tl < tf + 0.28:
            a = 1 - (tl - tf) / 0.28
            im = Image.blend(im, Image.new('RGB', im.size, (255, 244, 220)), 0.9 * a)
    elif si == 4:
        t_open, t_iso = 3.7, 5.74
        ap = smooth(prog(tl, t_open, 1.5))
        iso = smooth(prog(tl, t_iso, 2.3))
        bright = 0.20 + 0.28 * ap + 0.55 * iso
        grain = 0.01 + 0.06 * iso
        arr, _ = rphoto('sharp', tl, dur, bright=bright, grain=grain, seed=fr)
        d = paste_photo(im, arr)
        vals = [400, 800, 1600, 3200]
        iso_txt = str(vals[min(3, int(iso * 4))]) if tl >= t_iso else "400"
        hl = 2 if tl >= t_iso else (1 if tl >= t_open else None)
        readouts(d, "1/1000", "f/2.8" if ap > 0.5 else "f/8", iso_txt, hl, AMB)
        if tl < t_open:
            label(d, (56, 90), "! TOO DARK", REDS, 28, 1)
        elif tl > t_iso + 2.4:
            label(d, (56, 90), "ok EXPOSED", GRN, 28, 1)
    elif si == 5:
        a1, _ = rphoto('sharp', tl, dur, grain=0.075, seed=fr)
        a2, _ = rphoto('blur_all', tl, dur)
        a1 = cv2.resize(a1, (int(350 * SS), int(200 * SS)), interpolation=cv2.INTER_AREA)
        a2 = cv2.resize(a2, (int(350 * SS), int(200 * SS)), interpolation=cv2.INTER_AREA)
        pst(im, a1, (30, 110))
        pst(im, a2, (420, 110))
        d = SD(im)
        for xx in (30, 420):
            d.rectangle([xx - 2, 108, xx + 351, 311], outline=DIMA, width=2)
        sd(d, (30, 84), "GRAIN", 24)
        sd(d, (420, 84), "BLUR", 24)
        if tl > 0.9:
            label(d, (30, 350), "ok FIX IN EDIT", GRN, 24, 1)
        if tl > 1.66 + 0.5:
            label(d, (420, 350), "x CAN'T FIX", REDS, 24, 1)
        if tl > 2.68:
            sd(d, (400, 470), "SHARP > CLEAN", 34, (255, 230, 190), 'mm')
    else:
        lines = [("> try it friday night", 0.3), ("> comment SETTINGS", 1.66), ("sending setup.pdf ...", 2.68)]
        for k, (s, t0) in enumerate(lines):
            n = int(clamp((tl - t0) * 18, 0, len(s)))
            sd(d, (40, 110 + k * 76), s[:n], 36, AMB if k < 2 else DIMA)
            if 0 < n < len(s) or (n == len(s) and k == 1 and tl < 2.68):
                cx = 40 + mono(36).getlength(s[:n])
                if int(tl * 5) % 2 == 0:
                    d.rectangle([cx + 2, 110 + k * 76 - 18, cx + 22, 110 + k * 76 + 18], fill=AMB)
        if tl > 2.68:
            p = clamp((tl - 2.68) / 1.6)
            d.rectangle([40, 360, 760, 410], outline=AMB, width=3)
            d.rectangle([44, 364, 44 + 712 * p, 406], fill=AMB)
            sd(d, (400, 450), f"{int(p * 100)}%", 26, AMB, 'mm')
        if tl > 4.3:
            sd(d, (40, 520), "ok FULL SETUP SENT", 38, GRN)
    a = np.array(im, np.float32)
    h_, w_ = a.shape[:2]
    a[1::2] *= 0.88   # scanlines
    yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float32)
    v = 1 - 0.38 * (((xx / SS - 400) / 440) ** 2 + ((yy / SS - 300) / 340) ** 2)
    a *= np.clip(v, 0.3, 1)[:, :, None]
    a *= 1.0 + 0.03 * math.sin(gt * 55)
    a = np.clip(a, 0, 255)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    rgba = np.zeros((h_, w_, 4), np.uint8)
    rgba[:, :, :3] = a.astype(np.uint8)
    mask = np.zeros((h_, w_), np.uint8)
    k = lambda v_: int(round(v_ * SS))
    cv2.rectangle(mask, (k(30), 0), (k(770), h_), 255, -1)
    cv2.rectangle(mask, (0, k(30)), (w_, k(570)), 255, -1)
    for cx, cy in ((30, 30), (770, 30), (30, 570), (770, 570)):
        cv2.circle(mask, (k(cx), k(cy)), k(30), 255, -1, cv2.LINE_AA)
    rgba[:, :, 3] = cv2.GaussianBlur(mask, (0, 0), 1.2)
    return rgba




# ------------------------------------------------------------------ intro: Lightroom-type screen + person
_INTRO = {}


def intro_photo():
    if 'p' not in _INTRO:
        a, _ = rphoto('blur_all', 0.0, 1.0)
        _INTRO['p'] = a
    return _INTRO['p']


def slider(d, x, y, w, label_, val, col=AMB, vtxt=None):
    sd(d, (x, y), label_, 17, DIMA)
    d.line([x, y + 24, x + w, y + 24], fill=(90, 56, 24), width=4)
    kx = x + w * val
    d.line([x, y + 24, kx, y + 24], fill=col, width=4)
    d.ellipse([kx - 8, y + 16, kx + 8, y + 32], fill=col)
    sd(d, (x + w, y), vtxt if vtxt else f"{int((val - 0.5) * 200):+d}", 17, col, 'rm')


def intro_screen(tl, fr):
    im = Image.new('RGB', (int(800 * SS), int(600 * SS)), BGS)
    d = SD(im)
    d.rectangle([0, 0, 800, 36], fill=(44, 24, 10))
    sd(d, (14, 18), "PHOTO LAB", 20, AMB)
    for i, tname in enumerate(["LIBRARY", "DEVELOP", "MAP", "BOOK"]):
        x = 190 + i * 112
        sd(d, (x, 18), tname, 18, AMB if tname == "DEVELOP" else DIMA)
        if tname == "DEVELOP":
            d.line([x, 32, x + 90, 32], fill=AMB, width=3)
    # main image
    arr = cv2.resize(intro_photo(), (int(540 * SS), int(300 * SS)), interpolation=cv2.INTER_AREA)
    pst(im, arr, (20, 50))
    d.rectangle([18, 48, 561, 351], outline=DIMA, width=2)
    sd(d, (24, 366), "IMG_0412.CR3   1/125   f/2.8   ISO 3200", 17, DIMA)
    # right panel
    d.rectangle([575, 48, 790, 420], outline=(90, 56, 24), width=2)
    sd(d, (588, 66), "BASIC", 20, AMB)
    slider(d, 588, 96, 190, "EXPOSURE", 0.5)
    slider(d, 588, 150, 190, "CONTRAST", 0.5)
    sh = 0.35 + 0.60 * smooth(prog(tl, 0.7, 1.5))
    slider(d, 588, 204, 190, "SHARPNESS", sh, REDS if tl > 1.2 else AMB, f"{int(sh * 150)}")
    slider(d, 588, 258, 190, "CLARITY", 0.5)
    if tl > 2.0 and int(tl * 3) % 2 == 0:
        sd(d, (682, 340), "STILL BLURRY", 20, REDS, 'mm')
    # filmstrip
    d.rectangle([14, 396, 786, 590], outline=(90, 56, 24), width=2)
    base = intro_photo()
    for i in range(6):
        x = 28 + i * 126
        ry, rx = 40 + (i * 23) % 60, (i * 83) % 120
        crop = base[int(ry * SS): int((ry + 300) * SS), int(rx * SS): int((rx + 560) * SS)]
        th = cv2.resize(crop, (int(118 * SS), int(74 * SS)), interpolation=cv2.INTER_AREA)
        if i % 2:
            th = th[:, ::-1]
        pst(im, th, (x, 410))
        d.rectangle([x - 2, 408, x + 119, 486], outline=AMB if i == 2 else (90, 56, 24), width=3 if i == 2 else 2)
        if tl > 0.3 + i * 0.22:
            sd(d, (x + 8, 504), "x", 22, REDS)
            sd(d, (x + 34, 504), "BLUR", 17, REDS)
    sd(d, (30, 560), "6 / 6 FLAGGED", 18, DIMA)
    return im


def make_person(t):
    """Over-the-shoulder silhouette (seen from behind), hand on head. Returns (rgba, x0, y0) region."""
    ox, oy, rw, rh = 0, 760, 760, H - 760
    S = 2
    im = Image.new('L', (rw * S, rh * S), 0)
    d = ImageDraw.Draw(im)
    cx = 260
    sig = smooth((t - 0.9) / 0.5) * (1 - smooth((t - 1.9) / 0.7))
    bob = 5 * math.sin(t * 1.5) + 22 * sig
    rub = 9 * math.sin(t * 3.1) * smooth(t / 0.4)
    ex = -smooth((t - 2.5) / 1.0) * 700
    def P(x, y):
        return ((x + cx + ex - ox) * S, (y - oy) * S)
    def E(x0, y0, x1, y1):
        d.ellipse([*P(x0, y0), *P(x1, y1)], fill=255)
    sy = 12 * sig
    E(-440, 1530 + sy, 420, 2400)                       # shoulders / back
    d.polygon([P(-100, 1370 + bob), P(100, 1370 + bob), P(190, 1580 + sy), P(-190, 1580 + sy)], fill=255)   # neck
    E(-180, 1050 + bob, 180, 1445 + bob)                # head
    E(-178, 1215 + bob, -138, 1300 + bob)               # ears
    E(138, 1215 + bob, 178, 1300 + bob)
    # raised arm (left side): shoulder -> elbow -> hand on head
    sx, syy = -300, 1580 + sy
    elx, ely = -430, 1330
    hx, hy = -112 + rub, 1100 + bob
    d.line([*P(sx, syy), *P(elx, ely)], fill=255, width=int(92 * S))
    d.line([*P(elx, ely), *P(hx, hy)], fill=255, width=int(78 * S))
    E(elx - 46, ely - 46, elx + 46, ely + 46)
    E(hx - 58, hy - 52, hx + 58, hy + 52)
    for k in range(4):                                  # fingers gripping hair
        a = math.radians(-35 + k * 28)
        d.line([*P(hx, hy), *P(hx + 78 * math.cos(a) + 12, hy + 62 * math.sin(a) + 30)], fill=255, width=int(26 * S))
    a8 = np.array(im.resize((rw, rh), Image.LANCZOS), np.float32) / 255.0
    a8 = cv2.GaussianBlur(a8, (0, 0), 2.2)                # soft focus: it is in front of the screen
    rim = np.clip(a8 - np.roll(a8, -9, axis=1), 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), 3.0)
    xx = np.linspace(0, 1, rw, dtype=np.float32)[None, :]
    body = np.stack([0.045 + 0.05 * xx, 0.03 + 0.03 * xx, 0.03 + 0.02 * xx], 2) * np.ones((rh, 1, 1), np.float32)
    rgb = body + rim[:, :, None] * np.array([1.0, 0.55, 0.28], np.float32) * 1.2
    rgba = np.dstack([np.clip(rgb, 0, 1), a8])
    return rgba, ox, oy


# ------------------------------------------------------------------ post
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * (((_xx - W / 2) / (W * 0.78)) ** 2 + ((_yy - H / 2) / (H * 0.70)) ** 2)).clip(0.25, 1)[:, :, None].astype(np.float32)
_hy, _hx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
_diag = (_hx * math.cos(math.radians(-33)) + _hy * math.sin(math.radians(-33)))
WARM = np.array([1.07, 0.99, 0.88], np.float32)


def post(cv, t, rs):
    img = cv.astype(np.float32) / 255.0
    # slow drifting cloud-light over everything
    m = 1 + 0.04 * np.sin((_diag + t * 38) / 95.0)
    img *= cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]
    # bloom
    sm = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = np.clip(sm - 0.50, 0, None) * 1.8
    b1 = cv2.GaussianBlur(b, (0, 0), 6)
    b2 = cv2.GaussianBlur(b, (0, 0), 22)
    bl = cv2.resize(b1 * 0.8 + b2 * 1.1, (W, H), interpolation=cv2.INTER_LINEAR)
    img += bl
    img *= WARM
    img = np.clip(img, 0, 1.2)
    img = img / (1 + 0.35 * np.clip(img - 0.8, 0, None))
    img = np.clip(img, 0, 1)
    img = 0.65 * img + 0.35 * (img * img * (3 - 2 * img))
    img *= VIG
    g = rs.randn(H // 2, W // 2).astype(np.float32) * 0.012
    img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]
    fade = smooth(t / 0.5) * (1 - smooth((t - (TOTAL - 0.5)) / 0.5))
    img *= fade
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ frame
def which(t):
    if t < OFF:
        return -1
    si = int(np.searchsorted(STARTS, t, 'right') - 1)
    return min(si, len(SC) - 1)


SCREEN_Q = np.array([[-19, 119.5, 50.3], [19, 119.5, 50.3], [19, 91, 50.3], [-19, 91, 50.3]], float)
BEZEL_Q = np.array([[-24, 127, 50], [24, 127, 50], [24, 83, 50], [-24, 83, 50]], float)
WALL_Q = np.array([[-110, 260, 0], [110, 260, 0], [110, 20, 0], [-110, 20, 0]], float)
DESK_Q = np.array([[-120, 75, 0], [120, 75, 0], [120, 75, 85], [-120, 75, 85]], float)
KEY_Q = np.array([[-20, 75.8, 57], [20, 75.8, 57], [20, 75.8, 79], [-20, 75.8, 79]], float)
def _mscale(P, k=1.10):
    P = np.array(P, float).copy()
    P[:, 0] *= k
    P[:, 1] = 105 + (P[:, 1] - 105) * k
    return P


SCREEN_Q = _mscale(SCREEN_Q)
BEZEL_Q = _mscale(BEZEL_Q)
TXT_Q = np.array([[-38, 168, 0.4], [38, 168, 0.4], [38, 168 - 24.8, 0.4], [-38, 168 - 24.8, 0.4]], float)


def render(fr, ytxt=None):
    t = fr / FPS
    si = which(t)
    tl = t - (STARTS[si] if si >= 0 else 0.0)
    cam = Cam(t)
    cv = np.zeros((HS, WS, 3), np.uint8)
    quad(cv, cam, WALL, WALL_Q)
    quad(cv, cam, DESK, DESK_Q)
    if si >= 0:
        quad(cv, cam, wall_text(si, tl, SC[si].dur), TXT_Q, alpha=True)
    # monitor body (sides / top / dark back)
    body = (88, 80, 70)
    poly(cv, cam, _mscale([[-24, 127, 50], [24, 127, 50], [18, 122, 12], [-18, 122, 12]]), (150, 140, 120), (0, 1, 0))
    poly(cv, cam, _mscale([[-24, 127, 50], [-24, 83, 50], [-18, 88, 12], [-18, 122, 12]]), (100, 92, 78), (-1, 0, 0))
    poly(cv, cam, _mscale([[24, 127, 50], [24, 83, 50], [18, 88, 12], [18, 122, 12]]), (160, 148, 126), (1, 0, 0))
    quad(cv, cam, BEZEL, BEZEL_Q)
    quad(cv, cam, screen_scene(si, tl, SC[si].dur if si >= 0 else OFF, fr, t), SCREEN_Q, alpha=True)
    quad(cv, cam, (KEYB * 0.8).astype(np.uint8), KEY_Q)
    cv = cv2.resize(cv, (W, H), interpolation=cv2.INTER_AREA)
    if t < OFF:
        pr, px_, py_ = make_person(t)
        a = pr[:, :, 3:4]
        reg = cv[py_:py_ + pr.shape[0], px_:px_ + pr.shape[1]].astype(np.float32)
        cv[py_:py_ + pr.shape[0], px_:px_ + pr.shape[1]] = np.clip(reg * (1 - a) + pr[:, :, :3] * 255 * a, 0, 255).astype(np.uint8)
    out = post(cv, t, np.random.RandomState(fr))
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    d.text((W // 2, H - 70), f"{fr:04d} / {NFR:04d}", font=mono(26), fill=(255, 214, 170), anchor='mm')
    return np.array(im)


def render_range(args):
    a, b, path = args
    tmp = path[:-4] + '.part.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-c:v', 'libx264', '-crf', '10', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-g', '30', tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in range(a, b):
        p.stdin.write(render(fr).tobytes())
    p.stdin.close()
    rc = p.wait()
    if rc == 0:
        os.replace(tmp, path)
    return rc
