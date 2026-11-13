"""Place sfx-kit sounds (vetted commit 92780e7) on the Day 1 timeline using each sound's sync point."""
import json, os, subprocess, numpy as np, wave, sys
ROOT = os.environ.get("REEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
KIT = os.environ.get("SFX_KIT", f"{ROOT}/vendor/sfx-kit")
SR, DUR = 48000, 5.0
meta = {s["id"]: s for s in json.load(open(f"{KIT}/sounds.json"))["sounds"]}

# (event time s, sound id, gain dB, pan -1..1, optional max length s)
CUES = [
    (0.55, "whoosh-slow-soft",              -7,  0.0, None),   # camera push-in starts
    (0.80, "whoosh-fast-soft",             -13, -0.3, None),   # head shake
    (0.10, "pop-small-in",                 -13,  0.0, None),   # caption chunk 1
    (0.80, "pop-small-in",                 -13,  0.0, None),   # caption chunk 2
    (1.80, "notification-message-in-soft",  -4,  0.2, None),   # Claude notices him
    (1.95, "pop-small-in",                 -13,  0.2, None),   # caption chunk 3
    (3.15, "pop-small-in",                 -13,  0.2, None),   # caption chunk 4
    (3.95, "pop-small-in",                 -13,  0.2, None),   # caption chunk 5
    (4.67, "riser-1s-soft",                 -8,  0.0, None),   # riser ENDS on the fix
    (4.67, "shimmer-medium-sparkle",        -6,  0.0, None),   # photo turns clean
    (4.70, "tone-ding-high",               -10,  0.1, None),
]


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


N = int(SR * DUR)
mix = np.zeros((N, 2))
table = []
for t, sid, gain, pan, maxlen in CUES:
    s = meta[sid]; y = load(f"{KIT}/{s['file']}")
    if maxlen:
        n = int(maxlen * SR); y = y[:n]; f = int(0.12 * SR); y[-f:] *= np.linspace(1, 0, min(f, len(y)))[:len(y[-f:])][:, None]
    start = t - s["sync"]["at"]
    i = int(round(start * SR)); g = 10 ** (gain / 20)
    gl, gr = g * (1 - max(0, pan)), g * (1 + min(0, pan))
    if i < 0: y = y[-i:]; i = 0
    y = y[: N - i]
    mix[i:i + len(y), 0] += y[:, 0] * gl; mix[i:i + len(y), 1] += y[:, 1] * gr
    table.append((t, sid, round(start, 3), round(s["sync"]["at"], 3), gain))
    print(f"{t:5.2f}s  {sid:32s} start {start:6.3f}  sync@{s['sync']['at']:.3f}  {gain} dB")
peak = np.abs(mix).max(); print("peak before master:", round(float(peak), 3))
mix = mix / max(1.0, peak / 0.9)
with wave.open(f"{ROOT}/sfx_raw.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
json.dump([dict(t=a, id=b, start=c, sync=d, gain_db=e) for a, b, c, d, e in table], open(f"{ROOT}/cues_day1.json", "w"), indent=1)

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{ROOT}/sfx_raw.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=7",
                "-ar", "48000", f"{ROOT}/sfx_final.wav"], check=True)
print("wrote", f"{ROOT}/sfx_final.wav")
