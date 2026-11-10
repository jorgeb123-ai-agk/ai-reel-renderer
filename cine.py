"""Cinematic remake of Day 2 (shutter speed): a small lit 3D room rendered in software.
Pinhole camera, textured planes via homographies, baked window-blind light, CRT screen content drawn per frame,
bloom / grade / vignette / grain in post. Everything is computed per frame."""
import math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
HQ = os.environ.get('CINE_HQ') == '1'   # high-quality look: 2x supersampling, glass reflection, screen glow, wall prints, finer textures
SSF = 2.0 if HQ else 1.5   # 3D supersampling factor
SS = 2.0 if HQ else 1.5    # CRT screen texture scale (1.0 = 800x600)
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

    def polygon(self, xy, fill=None):
        self.d.polygon(self._xy(xy), fill=fill)

    def text(self, xy, s, font=None, fill=None, anchor=None):
        self.d.text(self._xy(xy), s, font=font, fill=fill, anchor=anchor)


def pst(im, arr, pos):
    im.paste(Image.fromarray(np.ascontiguousarray(arr)), (int(pos[0] * SS), int(pos[1] * SS)))


HERE = os.path.dirname(os.path.abspath(__file__))
DAY = os.environ.get('CINE_DAY', '2')   # '2' = shutter speed (approved reference), '1' = raw vs edit
PHOTO_PATH = os.environ.get('PHOTO', os.path.join(HERE, 'assets', 'photo.jpg'))
RAW_PATH = os.environ.get('RAW_PHOTO', os.path.join(HERE, 'assets', 'raw.jpg'))       # day 1: unedited RAW export
FINAL_PATH = os.environ.get('FINAL_PHOTO', os.path.join(HERE, 'assets', 'final.jpg'))  # day 1: the real final edit
BRAND = 'Jorgeb_photography'                      # printed on the monitor's top bezel in every scene
TITLE = os.environ.get('CINE_TITLE', 'RAW VS EDIT' if DAY == '1' else 'SHUTTER SPEED')   # video title, bottom bezel


class _Scene:
    def __init__(self, dur):
        self.dur = dur


# scene durations in seconds (problem, menu, bad result, fix, dark/ISO, trick, CTA).
# When voice clips exist, replace these with clip length + ~0.3 s tail.
SCENE_DURS = [3.91, 5.61, 5.61, 3.2, 9.01, 4.59, 5.27]
# day 1 (raw vs edit): hook, the raw, white balance, shadows, contrast+color, "that's it", reveal, CTA
D1_DURS = [3.2, 5.4, 5.8, 4.8, 4.1, 3.6, 4.2, 4.2]
if DAY == '1':
    SCENE_DURS = D1_DURS
SC = [_Scene(d) for d in SCENE_DURS]
INTRO = 3.6   # intro shot (person + editor) before the original scenes
OPEN = 4.2 if HQ else 0.0   # HQ: wide establishing shot of the desk, then a slow pan-in, before the intro
OFF = OPEN + INTRO
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
    (INTRO, (5, 114, 126), (0, 113, 45)),
    (INTRO + 3.91, (-3, 113, 118), (0, 112, 45)),
    (INTRO + 9.52, (2, 112, 112), (0, 112, 46)),
    (INTRO + 15.13, (-2, 112, 110), (0, 111, 46)),
    (INTRO + 18.33, (2, 112, 112), (0, 112, 46)),
    (INTRO + 27.34, (-3, 113, 118), (0, 113, 45)),
    (INTRO + 31.93, (0, 114, 122), (0, 113, 45)),
    (INTRO + 37.2, (0, 115, 134), (0, 114, 45)),
]
if DAY == '1':   # stretch the camera path over this video's length (keys after the intro keep their relative timing)
    _s = (STARTS[-1] - OFF) / 37.2
    KEYS = [KEYS[0]] + [(INTRO + (k[0] - INTRO) * _s, k[1], k[2]) for k in KEYS[1:]]
WIDE_P, WIDE_T = (-6, 110, 196), (-8, 105, 34)      # wide establishing view of the desk
if HQ:   # hold the wide shot ~1 s, ease in to the intro framing, then follow the normal path (shifted by OPEN)
    KEYS = [(0.0, WIDE_P, WIDE_T), (1.0, WIDE_P, WIDE_T)] + [(k[0] + OPEN, k[1], k[2]) for k in KEYS]
_kt = np.array([k[0] for k in KEYS])
_kv = np.array([list(k[1]) + list(k[2]) for k in KEYS], dtype=float)
_k0 = 2 if HQ else 1   # keys before this index (the wide establishing shot) are not calmed
_km = _kv[_k0:].mean(0)
_kv[_k0:] = _km + 0.6 * (_kv[_k0:] - _km)


def cam_path(t):
    t = clamp(t, 0, _kt[-1] - 1e-6)
    i = int(np.searchsorted(_kt, t, 'right') - 1)
    i = min(i, len(_kt) - 2)
    dt = _kt[i + 1] - _kt[i]
    s = (t - _kt[i]) / dt

    def tang(j):
        if HQ and j <= 2:
            return np.zeros(6)
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
    if alpha:   # mip-style prefilter: bring props close to their on-screen size before the perspective warp
        sw = max(np.hypot(*(pts[1] - pts[0])), np.hypot(*(pts[2] - pts[3])))
        sh_ = max(np.hypot(*(pts[3] - pts[0])), np.hypot(*(pts[2] - pts[1])))
        f = max(min(sw * 1.5 / tw, sh_ * 1.5 / th, 1.0), 0.1)
        if f < 0.75:
            tex = cv2.resize(tex, (max(int(tw * max(min(sw * 1.5 / tw, 1.0), 0.1)), 4), max(int(th * max(min(sh_ * 1.5 / th, 1.0), 0.1)), 4)), interpolation=cv2.INTER_AREA)
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


BLIND_STRIPES = False
FILM_GRAIN = False     # True brings back the film grain over the room
WALL_MOTTLE = False    # True brings back the blotchy plaster patches and fine grain on the wall


# ------------------------------------------------------------------ textures (baked once)
def blind_light(U, V, period, ang, phase=0.0, soft=0.07):
    # Window-blind shadow stripes are switched off: return the stripes' average level, so brightness stays
    # the same but the scene has soft, stripe-free light. Set BLIND_STRIPES = True to bring them back.
    if not BLIND_STRIPES:
        return np.full(np.broadcast(U, V).shape, 0.42, np.float32)
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
    if not WALL_MOTTLE:
        noise = np.full_like(noise, 0.5)
    base = np.array([176, 122, 88], np.float32)[None, None, :] * (0.93 + 0.14 * noise)[:, :, None]
    base *= (1 - 0.55 * np.clip((Y - 80) / 180, 0, 1))[:, :, None]
    band = blind_light(X, Y, 34.0, -33, 0.15, 0.24)
    env = 0.40 + 0.60 * sstep(-0.6, 0.9, (-(X + 25) * 0.012 + (Y - 120) * 0.0 + 0.55 - np.abs(Y - 130) * 0.004))
    light = 0.70 + 0.28 * band * env
    # soft monitor shadow behind it
    ms = np.exp(-(((X - 6) / 34) ** 4 + ((Y - 104) / 30) ** 4))
    light *= (1 - 0.55 * ms)
    warm = np.stack([np.ones_like(light), 0.80 + 0.2 * (1 - light), 0.55 + 0.4 * (1 - light)], 2)
    tex = base * light[:, :, None] * (0.9 + 0.1 * warm) * 1.18
    tex = cv2.GaussianBlur(tex, (0, 0), 2.2)  # depth of field
    if HQ and WALL_MOTTLE:   # fine plaster grain that survives the depth-of-field blur
        fine = cv2.GaussianBlur(np.random.RandomState(9).randn(*tex.shape[:2]).astype(np.float32), (0, 0), 0.8)
        tex = tex * (1 + 0.05 * fine)[:, :, None]
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
    k = 2 if HQ else 1          # HQ draws the bezel texture at 2x
    pxcm = 10 * k
    w, h = 48 * pxcm, 44 * pxcm
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([168, 154, 130], np.float32)[None, None, :] * (0.90 + 0.10 * (1 - yy / h))[:, :, None]
    band = blind_light(xx * 0.1 / k, yy * 0.1 / k, 17.0, -33, 0.0, 0.1)
    base *= (0.93 + 0.10 * band)[:, :, None]
    if HQ:
        rs_ = np.random.RandomState(5)
        base *= (1 + (0.022 if FILM_GRAIN else 0.0) * cv2.GaussianBlur(rs_.randn(h, w).astype(np.float32), (0, 0), 0.7))[:, :, None]   # plastic grain
        mk = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mk).rounded_rectangle([40 * k, 55 * k, 440 * k, 372 * k], radius=(54 if HQ else 34) * k, fill=255)
        mk = np.array(mk, np.float32) / 255.0
        ao = cv2.GaussianBlur(mk, (0, 0), 16 * k)
        base *= (1 - 0.42 * ao * (1 - mk))[:, :, None]                      # soft shadow where the screen is recessed
        edge = np.exp(-(yy / (3.0 * k)) ** 2) + np.exp(-(xx / (3.0 * k)) ** 2) * 0.6   # light catching the top / left edges
        base *= (1 + 0.07 * edge)[:, :, None]
        spec = np.exp(-(((xx / w) * 0.9 + (yy / h) * 0.5 - 0.30) / 0.05) ** 2) * (yy / h < 0.12)
        base *= (1 + 0.05 * spec)[:, :, None]
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, 'RGBA')
    d.rounded_rectangle([40 * k, 55 * k, 440 * k, 372 * k], radius=(54 if HQ else 34) * k, fill=(26, 22, 20), outline=(120, 108, 92), width=3 * k)
    d.rounded_rectangle([46 * k, 61 * k, 434 * k, 366 * k], radius=(50 if HQ else 30) * k, outline=(0, 0, 0, 120), width=5 * k)
    d.ellipse([382 * k, 390 * k, 398 * k, 406 * k], fill=(120, 255, 120))
    for j in range(6):
        d.line([(60 + j * 18) * k, 396 * k, (60 + j * 18) * k, 414 * k], fill=(120, 108, 92), width=3 * k)
    ttl, fs = TITLE.upper(), 15
    while fs > 9 and mono(fs).getlength(ttl) > 190:   # keep it between the vent slots and the power light
        fs -= 1
    d.text((240 * k, 400 * k), ttl, font=mono(fs * k), fill=(104, 94, 80), anchor='mm')
    d.text((240 * k, 27 * k), BRAND, font=mono(15 * k), fill=(104, 94, 80), anchor='mm')   # same font/colour as the bottom label
    if HQ:   # rounded outer corners (alpha), so the whole monitor reads rounder
        al = Image.new('L', (w, h), 0)
        ImageDraw.Draw(al).rounded_rectangle([0, 0, w - 1, h - 1], radius=int(6.0 * pxcm), fill=255)
        al = cv2.GaussianBlur(np.array(al), (0, 0), 1.1 * k)
        return np.dstack([np.array(im.convert('RGB')), al])
    return np.array(im)


def make_keyboard():
    k = 2 if HQ else 1
    w, h = 400 * k, 220 * k
    im = Image.new('RGB', (w, h), (190, 176, 150))
    d = ImageDraw.Draw(im)
    legends = ["`1234567890-=<", "TQWERTYUIOP[]\\", "CASDFGHJKL;'  ", "SZXCVBNM,./^> "]
    for r in range(5):
        n = 14 if r < 4 else 1
        for c in range(n):
            if r == 4:
                box = [90 * k, (22 + r * 40) * k, 310 * k, (54 + r * 40) * k]
                if HQ:
                    d.rounded_rectangle([box[0], box[1] + 3 * k, box[2], box[3] + 3 * k], 5 * k, fill=(150, 138, 114))
                d.rounded_rectangle(box, 5 * k, fill=(214, 202, 176))
                if HQ:
                    d.line([box[0] + 6 * k, box[1] + 2 * k, box[2] - 6 * k, box[1] + 2 * k], fill=(232, 222, 198), width=k)
                break
            x0 = (10 + c * 27.5 + (r % 2) * 5) * k
            box = [x0, (18 + r * 40) * k, x0 + 24 * k, (50 + r * 40) * k]
            if HQ:
                d.rounded_rectangle([box[0], box[1] + 3 * k, box[2], box[3] + 3 * k], 5 * k, fill=(150, 138, 114))   # key side / shadow
            d.rounded_rectangle(box, 5 * k, fill=(214, 202, 176))
            if HQ:
                d.line([box[0] + 5 * k, box[1] + 2 * k, box[2] - 5 * k, box[1] + 2 * k], fill=(232, 222, 198), width=k)  # top-edge highlight
                ch_ = legends[r][c] if c < len(legends[r]) else ' '
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - k), ch_, font=mono(9 * k), fill=(128, 114, 94), anchor='mm')
    a = np.array(im, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = blind_light(xx * 0.1 / k, yy * 0.1 / k, 15.0, 18, 0.2, 0.1)
    a *= (0.90 + 0.12 * band)[:, :, None]
    return np.clip(a, 0, 255).astype(np.uint8)


WALL, DESK, BEZEL, KEYB = make_wall(), make_desk(), make_bezel(), make_keyboard()


_GLASS = {}
def glass_overlay(h_, w_):
    """HQ only: soft window reflection across the CRT glass (screen-space, additive)."""
    if (h_, w_) not in _GLASS:
        yy, xx = np.mgrid[0:h_, 0:w_].astype(np.float32)
        u, v = xx / w_, yy / h_
        band = np.exp(-((u * 0.75 + v * 0.55 - 0.50) / 0.09) ** 2) * 0.55 + np.exp(-((u * 0.75 + v * 0.55 - 0.78) / 0.04) ** 2) * 0.35
        corner = np.exp(-(((u - 0.08) / 0.35) ** 2 + ((v - 0.06) / 0.30) ** 2))
        g = (0.050 * band + 0.035 * corner)[:, :, None] * np.array([255, 236, 208], np.float32)[None, None, :]
        _GLASS[(h_, w_)] = g
    return _GLASS[(h_, w_)]


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
# day 1 wall headlines (line, start time within the scene, seconds); timed to the narration words
D1_WLINES = [
    [("THIS PHOTO", 0.25), ("LOOKED TRASH.", 0.95), ("OUT OF CAMERA.", 1.9)],
    [("THE RAW.", 0.25), ("FLAT. DARK.", 2.4), ("COLORS OFF.", 3.9)],
    [("1. WHITE BALANCE", 0.25), ("FIX THE COLOR.", 3.0), ("REAL JERSEYS.", 4.3)],
    [("2. SHADOWS UP", 0.25), ("SEE HIS FACE.", 2.6)],
    [("3. CONTRAST", 0.25), ("+ COLOR.", 1.9), ("IT POPS.", 2.9)],
    [("THAT'S IT.", 0.25), ("FOUR SLIDERS.", 1.0), ("TWO MINUTES.", 2.0)],
    [("NOW IT LOOKS", 0.25), ("LIKE THE", 1.3), ("MOMENT FELT.", 1.9)],
    [("SAVE THIS.", 0.25), ("FOR FLAT EDITS.", 1.6)],
]
if DAY == '1':
    WLINES = D1_WLINES
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
_BASE_W, _BASE_H = (2160, 1440) if HQ else (1440, 960)


_LUMW = np.array([0.299, 0.587, 0.114], np.float32)


def _d1_grade(a, GP, wb=1.0, sh=1.0, sat=1.0):
    """The three global edits that carry the RAW toward the FINAL (see fit_grade.py). a: float RGB in 0..1."""
    a = np.clip(a * (1 + (GP['G'] - 1) * wb).astype(np.float32), 0, 1)
    if sh > 0:
        l = a @ _LUMW
        l2 = l + sh * (np.interp(l, GP['knots'], GP['ys']) - l)
        a = np.clip(a * (l2 / np.maximum(l, 1e-3))[..., None], 0, 1)
    if sat > 0:
        l = (a @ _LUMW)[..., None]
        a = np.clip(l + (a - l) * (1 + (float(GP['Sat']) - 1) * sat), 0, 1)
    return a


def _rp_base():
    if DAY == '1' and 'raw' not in _RP:
        ld = lambda p: cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (_BASE_W, _BASE_H),
                                  interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        raw, fin = ld(RAW_PATH), ld(FINAL_PATH)
        GP = np.load(os.path.join(HERE, 'grade_params.npz'))
        _RP['GP'] = GP
        _RP['raw'], _RP['final'], _RP['sharp'] = raw, fin, raw
        # whatever the 3 global steps cannot reach (local edits) is faded in with the last step
        _RP['res'] = fin - _d1_grade(raw, GP)
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


def rphoto_d1(tl, dur, wb=0.0, sh=0.0, sat=0.0, res=0.0, src='raw'):
    """Day 1: the RAW with the edit steps applied to the given amounts (0..1 each), or the real FINAL (src='final').
    Same slow push-in as rphoto. Returns a 720x400 (x SS) uint8 crop."""
    R = _rp_base()
    z = 1.0 + 0.045 * clamp(tl / max(dur, 0.1))
    f = _BASE_W / 1440.0
    cw, ch = _BASE_W / z, 800.0 * f / z
    x0 = (_BASE_W - cw) / 2
    y0 = 80 * f + (800 * f - ch) / 2
    OW, OH = int(PWD * SS), int(PHT * SS)
    M = np.float32([[OW / cw, 0, -x0 * OW / cw], [0, OH / ch, -y0 * OH / ch]])
    fl = cv2.INTER_AREA if z < 1.01 else cv2.INTER_LINEAR
    warp = lambda s: cv2.warpAffine(s, M, (OW, OH), flags=fl, borderMode=cv2.BORDER_REFLECT)
    a = warp(R[src])
    if src == 'raw' and (wb > 0 or sh > 0 or sat > 0 or res > 0):
        a = _d1_grade(a, R['GP'], wb, sh, sat)
        if res > 0:
            a = np.clip(a + warp(R['res']) * res, 0, 1)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ CRT screen content (800x600)
TX0 = 40 if HQ else 14   # left/right margin of the app bar text (HQ corners are rounder, so keep text inside them)
AMB, DIMA, REDS, GRN = (255, 176, 66), (150, 96, 40), (255, 92, 64), (150, 255, 120)
BGS = (20, 11, 7)


def sd(d, xy, s, sz=22, fill=AMB, anchor='lm'):
    d.text(xy, s, font=mono(int(round(sz * SS))), fill=fill, anchor=anchor)


def screen_base(fr):
    im = Image.new('RGB', (int(800 * SS), int(600 * SS)), BGS)
    d = SD(im)
    d.rectangle([0, 0, 800, 36], fill=(44, 24, 10))
    sd(d, (TX0, 18), "PHOTO_LAB v2.1" if DAY == '1' else "SHUTTER_PRO v2.1", 22)
    sd(d, (800 - TX0, 18), f"FRM {fr:04d}", 20, DIMA, 'rm')
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


# ------------------------------------------------------------------ day 1 (raw vs edit) screen content
def d1_state(si, tl):
    """Edit amounts (wb, shadows, saturation, residual) for scene si at scene time tl, plus which slider is live."""
    if si <= 1:
        return 0.0, 0.0, 0.0, 0.0, None
    if si == 2:
        return smooth(prog(tl, 1.4, 1.3)), 0, 0, 0, (0 if 1.3 < tl < 2.9 else None)
    if si == 3:
        return 1.0, smooth(prog(tl, 1.0, 1.5)), 0, 0, (1 if 0.9 < tl < 2.6 else None)
    if si == 4:
        res = smooth(prog(tl, 0.3, 1.6))
        sat = smooth(prog(tl, 1.9, 1.1))
        return 1.0, 1.0, sat, res, (2 if 0.2 < tl < 1.95 else (3 if 1.85 < tl < 3.1 else None))
    return 1.0, 1.0, 1.0, 1.0, None


D1_SLIDERS = [("WHITE BAL", lambda wb, sh, sa, rs: f"{5200 + int(450 * wb)}K", lambda wb, sh, sa, rs: 0.5 + 0.30 * wb),
              ("SHADOWS", lambda wb, sh, sa, rs: f"{int(42 * sh):+d}", lambda wb, sh, sa, rs: 0.5 + 0.30 * sh),
              ("CONTRAST", lambda wb, sh, sa, rs: f"{int(24 * rs):+d}", lambda wb, sh, sa, rs: 0.5 + 0.30 * rs),
              ("COLOR", lambda wb, sh, sa, rs: f"{int(22 * sa):+d}", lambda wb, sh, sa, rs: 0.5 + 0.30 * sa)]


def edit_strip(d, st, live=None, flash=None, col=AMB):
    wb, sh, sa, rs, _ = st
    for i, (nm, vt, fr_) in enumerate(D1_SLIDERS):
        x0 = 40 + i * 182
        on = (live == i) or (flash is not None and flash >= i)
        d.rectangle([x0, 478, x0 + 168, 574], fill=col if on else (36, 20, 9), outline=col if on else DIMA, width=2)
        sd(d, (x0 + 10, 495), nm, 16, BGS if on else DIMA)
        sd(d, (x0 + 10, 524), vt(wb, sh, sa, rs), 28, BGS if on else AMB)
        k = x0 + 12 + 144 * fr_(wb, sh, sa, rs)
        d.line([x0 + 12, 558, x0 + 156, 558], fill=BGS if on else (90, 56, 24), width=4)
        d.ellipse([k - 7, 551, k + 7, 565], fill=BGS if on else AMB)


def hist_box(d, arr, x=588, y=58, w=162, h=64):
    """Small luminance histogram of the photo on screen, drawn over its top-right corner."""
    l = arr.astype(np.float32) @ _LUMW
    hh, _ = np.histogram(l, bins=36, range=(0, 255))
    hh = np.sqrt(hh / hh.max())
    d.rectangle([x, y, x + w, y + h], fill=(14, 8, 5), outline=DIMA, width=1)
    bw = (w - 8) / 36
    for i, v in enumerate(hh):
        d.rectangle([x + 4 + i * bw, y + h - 4 - v * (h - 12), x + 4 + (i + 1) * bw - 1, y + h - 4], fill=AMB)
    sd(d, (x + 6, y + 9), "HIST", 12, DIMA)


def day1_content(si, tl, dur, fr):
    if si == -1:
        return intro_screen(tl, fr)
    im, d = screen_base(fr)
    st = d1_state(si, tl)
    wb, sh, sa, rs, live = st
    if si == 6:   # reveal: RAW | FINAL divider sweeps right -> left, ending on the real final edit
        a_raw = rphoto_d1(tl, dur)
        a_fin = rphoto_d1(tl, dur, src='final')
        dv = 1 - smooth(prog(tl, 0.2, 1.9))
        cx = int(round(dv * a_raw.shape[1]))
        arr = np.ascontiguousarray(np.concatenate([a_raw[:, :cx], a_fin[:, cx:]], 1))
    else:
        arr = rphoto_d1(tl, dur, wb, sh, sa, rs)
    d = paste_photo(im, arr)
    sd(d, (52, 66), "IMG_0412.CR3", 18, (255, 220, 170))
    if si == 6:
        px = 40 + dv * 720
        d.line([px, 50, px, 450], fill=AMB, width=3)
        d.ellipse([px - 9, 240, px + 9, 258], fill=AMB)
        if dv > 0.12:
            label(d, (56, 424), "RAW", DIMA, 20, 1)
        if dv < 0.88:
            label(d, (640, 424), "FINAL", GRN, 20, 1)
        if tl > 2.3:
            sd(d, (400, 520), "RAW  >  FINAL", 34, (255, 230, 190), 'mm')
            sd(d, (400, 560), "PLAYOFF NIGHT", 20, DIMA, 'mm')
        return im
    hist_box(d, arr)
    flash = None
    if si == 5:
        flash = sum(1 for k in range(4) if tl > 1.0 + 0.25 * k) - 1
        flash = flash if flash >= 0 else None
    edit_strip(d, st, live, flash)
    if si == 0:
        if tl > 0.6:
            label(d, (56, 90), "x UNEDITED RAW", REDS, 22, 1)
    elif si == 1:
        for k, (t0, s) in enumerate(((2.5, "x FLAT"), (3.3, "x DARK"), (4.0, "x COLORS OFF"))):
            if tl > t0:
                label(d, (56, 90 + 44 * k), s, REDS, 22, 1)
    elif si == 2:
        if tl > 4.3:
            label(d, (56, 90), "ok TRUE COLOR", GRN, 22, 1)
    elif si == 3:
        if tl > 3.6:
            label(d, (56, 90), "ok FACES VISIBLE", GRN, 22, 1)
    elif si == 4:
        if tl > 3.0:
            label(d, (56, 90), "ok IT POPS", GRN, 22, 1)
    elif si == 5:
        if tl > 2.0:
            sd(d, (590, 150), "EDIT TIME", 16, DIMA)
            sd(d, (590, 178), "02:00", 34, (255, 230, 190))
    else:   # CTA terminal
        d.rectangle([0, 37, 800, 600], fill=BGS)
        lines = [("> save this post", 0.3), ("> for flat edits", 1.4), ("saving ...", 2.2)]
        for k, (s, t0) in enumerate(lines):
            n = int(clamp((tl - t0) * 18, 0, len(s)))
            sd(d, (40, 110 + k * 76), s[:n], 36, AMB if k < 2 else DIMA)
            if 0 < n < len(s) or (n == len(s) and k == 1 and tl < 2.2):
                cx = 40 + mono(36).getlength(s[:n])
                if int(tl * 5) % 2 == 0:
                    d.rectangle([cx + 2, 110 + k * 76 - 18, cx + 22, 110 + k * 76 + 18], fill=AMB)
        if tl > 2.6:
            p = smooth(prog(tl, 2.6, 0.4))
            ys = 330 - 20 * (1 - p)
            d.polygon([360, ys, 440, ys, 440, ys + 130, 400, ys + 96, 360, ys + 130], fill=AMB)
        if tl > 3.0:
            sd(d, (400, 520), "ok SAVED", 38, GRN, 'mm')
    return im


def screen_scene(si, tl, dur, fr, gt):
    if DAY == '1':
        im = day1_content(si, tl, dur, fr)
        d = SD(im)
    elif si == -1:
        im = intro_screen(tl, fr)
        d = SD(im)
    else:
        im, d = screen_base(fr)
    ph = tl * 9.0
    if DAY == '1' or si == -1:
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
    # (screen brightness flicker removed 2026-11-09 at the user's request)
    if HQ:
        a = a + glass_overlay(h_, w_)
    a = np.clip(a, 0, 255)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    rgba = np.zeros((h_, w_, 4), np.uint8)
    rgba[:, :, :3] = a.astype(np.uint8)
    mask = np.zeros((h_, w_), np.uint8)
    k = lambda v_: int(round(v_ * SS))
    R_ = 48 if HQ else 30   # corner radius of the glass (800x600 space); bigger = rounder
    cv2.rectangle(mask, (k(R_), 0), (k(800 - R_), h_), 255, -1)
    cv2.rectangle(mask, (0, k(R_)), (w_, k(600 - R_)), 255, -1)
    for cx, cy in ((R_, R_), (800 - R_, R_), (R_, 600 - R_), (800 - R_, 600 - R_)):
        cv2.circle(mask, (k(cx), k(cy)), k(R_), 255, -1, cv2.LINE_AA)
    rgba[:, :, 3] = cv2.GaussianBlur(mask, (0, 0), 1.2)
    return rgba




# ------------------------------------------------------------------ intro: Lightroom-type screen + person
_INTRO = {}


def intro_photo():
    if 'p' not in _INTRO:
        a = rphoto_d1(0.0, 1.0) if DAY == '1' else rphoto('blur_all', 0.0, 1.0)[0]
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
    sd(d, (TX0, 18), "PHOTO LAB", 20, AMB)
    for i, tname in enumerate(["LIBRARY", "DEVELOP", "MAP", "BOOK"]):
        x = 190 + i * 112
        sd(d, (x, 18), tname, 18, AMB if tname == "DEVELOP" else DIMA)
        if tname == "DEVELOP":
            d.line([x, 32, x + 90, 32], fill=AMB, width=3)
    # main image
    arr = cv2.resize(intro_photo(), (int(540 * SS), int(300 * SS)), interpolation=cv2.INTER_AREA)
    pst(im, arr, (20, 50))
    d.rectangle([18, 48, 561, 351], outline=DIMA, width=2)
    sd(d, (24, 366), "IMG_0412.CR3   RAW   UNEDITED" if DAY == '1' else "IMG_0412.CR3   1/125   f/2.8   ISO 3200", 17, DIMA)
    # right panel
    d.rectangle([575, 48, 790, 420], outline=(90, 56, 24), width=2)
    sd(d, (588, 66), "BASIC", 20, AMB)
    if DAY == '1':
        ex = 0.5 + 0.22 * smooth(prog(tl, 0.7, 1.5))
        slider(d, 588, 96, 190, "EXPOSURE", ex, REDS if tl > 1.2 else AMB)
        slider(d, 588, 150, 190, "CONTRAST", 0.5)
        slider(d, 588, 204, 190, "SHADOWS", 0.5)
        slider(d, 588, 258, 190, "COLOR", 0.5)
        if tl > 2.0 and int(tl * 3) % 2 == 0:
            sd(d, (682, 340), "STILL FLAT", 20, REDS, 'mm')
    else:
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
            sd(d, (x + 34, 504), "FLAT" if DAY == '1' else "BLUR", 17, REDS)
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


_CA = None


def post(cv, t, rs, gm=None):   # gm: 0..1 mask of the CRT screen, which gets no film grain
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
    if HQ:   # tiny lens fringing toward the corners, like a real lens
        global _CA
        if _CA is None:
            yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
            _CA = [(((xx_ - W / 2) * s_ + W / 2).astype(np.float32), ((yy_ - H / 2) * s_ + H / 2).astype(np.float32)) for s_ in (1.0009, 0.9991)]
        img[:, :, 0] = cv2.remap(img[:, :, 0], _CA[0][0], _CA[0][1], cv2.INTER_LINEAR)
        img[:, :, 2] = cv2.remap(img[:, :, 2], _CA[1][0], _CA[1][1], cv2.INTER_LINEAR)
    g = rs.randn(H // 2, W // 2).astype(np.float32) * (0.012 if FILM_GRAIN else 0.0)
    gr = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    if gm is not None:
        gr = gr * (1 - gm)
    img += gr[:, :, None]
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
    tl = t - (STARTS[si] if si >= 0 else OPEN)
    tl = max(tl, 0.0) if si < 0 else tl
    cam = Cam(t)
    cv = np.zeros((HS, WS, 3), np.uint8)
    quad(cv, cam, WALL, WALL_Q)
    if HQ:
        room.draw_room_back(cv, cam)
    quad(cv, cam, DESK, DESK_Q)
    if HQ:
        room.draw_desk_set(cv, cam)
    if si >= 0:
        quad(cv, cam, wall_text(si, tl, SC[si].dur), TXT_Q, alpha=True)
    # monitor body (sides / top / dark back)
    if HQ:
        room.draw_monitor_body_hq(cv, cam)
    else:
        poly(cv, cam, _mscale([[-24, 127, 50], [24, 127, 50], [18, 122, 12], [-18, 122, 12]]), (150, 140, 120), (0, 1, 0))
        poly(cv, cam, _mscale([[-24, 127, 50], [-24, 83, 50], [-18, 88, 12], [-18, 122, 12]]), (100, 92, 78), (-1, 0, 0))
        poly(cv, cam, _mscale([[24, 127, 50], [24, 83, 50], [18, 88, 12], [18, 122, 12]]), (160, 148, 126), (1, 0, 0))
    quad(cv, cam, BEZEL, BEZEL_Q, alpha=HQ)
    scr = screen_scene(si, tl, SC[si].dur if si >= 0 else INTRO, fr, t)
    quad(cv, cam, scr, SCREEN_Q, alpha=True)
    if HQ:
        room.draw_desk_front(cv, cam)
    else:
        quad(cv, cam, (KEYB * 0.8).astype(np.uint8), KEY_Q)
    cv = cv2.resize(cv, (W, H), interpolation=cv2.INTER_AREA)
    if HQ:   # the lit screen spills warm light onto the bezel, keyboard and wall (never onto the screen itself)
        m_ = scr[:, :, 3] > 128
        if m_.any():
            col = scr[:, :, :3][m_].astype(np.float32).mean(0) / 255.0
            pts, _ = cam.proj(SCREEN_Q)
            pts = pts / SSF
            sm = np.zeros((H // 8, W // 8, 3), np.float32)
            cv2.fillConvexPoly(sm, np.round(pts / 8 * 16).astype(np.int32), tuple(float(c) for c in col), cv2.LINE_AA, 4)
            glow = cv2.GaussianBlur(sm, (0, 0), 4) * 0.55 + cv2.GaussianBlur(sm, (0, 0), 11) * 0.9
            hole = cv2.GaussianBlur(np.clip(sm.max(2, keepdims=True), 0, 1)[:, :, 0], (0, 0), 0.6)
            glow = cv2.resize(glow * (1 - np.clip(hole * 1.5, 0, 1))[:, :, None], (W, H), interpolation=cv2.INTER_LINEAR)
            cv = np.clip(cv.astype(np.float32) + glow * 255 * 0.10, 0, 255).astype(np.uint8)
    if OPEN <= t < OFF:
        pr, px_, py_ = make_person(t - OPEN)
        a = pr[:, :, 3:4]
        reg = cv[py_:py_ + pr.shape[0], px_:px_ + pr.shape[1]].astype(np.float32)
        cv[py_:py_ + pr.shape[0], px_:px_ + pr.shape[1]] = np.clip(reg * (1 - a) + pr[:, :, :3] * 255 * a, 0, 255).astype(np.uint8)
    gm = np.zeros((H, W), np.uint8)   # the TV screen is kept grain-free
    pts, _ = cam.proj(SCREEN_Q)
    cv2.fillConvexPoly(gm, np.round(pts / SSF * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
    gm = cv2.GaussianBlur(gm, (0, 0), 2.0).astype(np.float32) / 255.0
    out = post(cv, t, np.random.RandomState(fr), gm)
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


import room  # noqa: E402  (HQ room props; imported last because room.py uses names defined above)
