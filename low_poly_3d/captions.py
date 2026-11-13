import os, sys, glob, subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get("REEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/frames"        # one PNG per frame at the final fps (540x960)
OUT = f"{ROOT}/final"
os.makedirs(OUT, exist_ok=True)
FPS = 24
NAME = os.environ.get("REEL_NAME", "day1_reel")
SCALE = 2                      # 540x960 -> 1080x1920

# word timings in seconds: (word, start, end, speaker)
G, C = "guy", "claude"
WORDS = [
    ("WHY", 0.10, 0.40, G), ("ARE", 0.40, 0.60, G), ("MY", 0.60, 0.80, G),
    ("PHOTOS", 0.80, 1.20, G), ("GRAINY?", 1.20, 1.75, G),
    ("THEY'RE", 1.95, 2.45, C), ("FINE.", 2.45, 3.00, C),
    ("RAISE", 3.15, 3.45, C), ("EXPOSURE,", 3.45, 3.95, C),
    ("ADD", 3.95, 4.15, C), ("NOISE", 4.15, 4.50, C), ("REDUCTION.", 4.50, 5.00, C),
]
# chunks = indices of WORDS shown together
CHUNKS = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 12)]

STYLE = {
    G: dict(name="YOU", hi=(255, 226, 60), tag=(255, 226, 60), tagtxt=(20, 20, 20)),
    C: dict(name="CLAUDE", hi=(255, 140, 40), tag=(255, 140, 40), tagtxt=(20, 20, 20)),
}


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


BASE = 92
frames = sorted(glob.glob(f"{SRC}/*.png"))
print(len(frames), "frames")
for i, fp in enumerate(frames):
    t = i / FPS
    im = Image.open(fp).convert("RGB").resize((540 * SCALE, 960 * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    W, H = im.size
    # pick chunk visible now
    cur = None
    for ci, (a, b) in enumerate(CHUNKS):
        start = WORDS[a][1]
        end = CHUNKS[ci + 1][0] and WORDS[CHUNKS[ci + 1][0]][1] if ci + 1 < len(CHUNKS) else 5.0
        if start - 0.05 <= t < end:
            cur = (a, b); break
    if cur:
        a, b = cur
        spk = WORDS[a][3]
        st = STYLE[spk]
        ws = WORDS[a:b]
        # layout: measure at base size
        gap = 28
        f0 = font(BASE)
        tot0 = sum(f0.getlength(w[0]) for w in ws) + gap * (len(ws) - 1)
        fit = min(1.0, (W * 0.88) / (tot0 * 1.1))
        bsz = int(BASE * fit)
        f0 = font(bsz)
        widths = [f0.getlength(w[0]) for w in ws]
        total = sum(widths) + gap * (len(ws) - 1)
        x = (W - total) / 2
        y = int(H * 0.70)
        for (word, s0, e0, sp), wd in zip(ws, widths):
            active = s0 <= t < e0
            spoken = t >= e0
            sz = bsz
            if active:
                sz = int(bsz * (1 + 0.14 * max(0.0, 1 - (t - s0) / 0.12)))
            ft = font(sz)
            wl = ft.getlength(word)
            cx = x + wd / 2
            # highlight box behind the active word
            if active:
                pad = 16
                d.rounded_rectangle([cx - wl / 2 - pad, y - sz * 0.62 - pad * 0.4, cx + wl / 2 + pad, y + sz * 0.5 + pad * 0.4], radius=22, fill=st["hi"])
                col, stroke = (20, 20, 20), None
                d.text((cx, y), word, font=ft, fill=col, anchor="mm")
            else:
                col = (255, 255, 255) if (spoken or True) else (255, 255, 255)
                d.text((cx, y), word, font=ft, fill=col, anchor="mm", stroke_width=9, stroke_fill=(0, 0, 0))
            x += wd + gap
        # speaker tag
        tf = font(46)
        tw = tf.getlength(st["name"])
        tx0 = (W - tw) / 2
        ty = y - BASE * 1.35
        d.rounded_rectangle([tx0 - 26, ty - 34, tx0 + tw + 26, ty + 34], radius=30, fill=st["tag"])
        d.text((W / 2, ty), st["name"], font=tf, fill=st["tagtxt"], anchor="mm")
    im.save(f"{OUT}/c_{i:04d}.png")

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{OUT}/c_%04d.png",
                "-vf", "vignette=PI/5,noise=alls=5:allf=t,eq=saturation=1.08:contrast=1.04",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", f"{ROOT}/{NAME}_silent.mp4"], check=True)
print("video written")

audio = f"{ROOT}/sfx_final.wav"
if os.path.exists(audio):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{ROOT}/{NAME}_silent.mp4", "-i", audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", f"{ROOT}/{NAME}.mp4"], check=True)
    print("video with sound written:", f"{ROOT}/{NAME}.mp4")
