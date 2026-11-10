import os, subprocess, wave, multiprocessing as mp
import numpy as np
import cine, sfx
from cine import *

def run(c):
    subprocess.run(c, check=True)

def tick(rs):
    n = int(0.05 * sfx.SR)
    k = np.arange(n) / sfx.SR
    f = rs.uniform(1500, 2300)
    y = rs.randn(n) * np.exp(-k * 160) * 0.5 + np.sin(2 * np.pi * f * k) * np.exp(-k * 220) * 0.5
    return np.convolve(y, np.ones(3) / 3, 'same') * 0.5

def dial(rs):
    n = int(0.12 * sfx.SR)
    k = np.arange(n) / sfx.SR
    y = np.zeros(n)
    for off, amp in ((0.0, 1.0), (0.045, 0.6)):
        i = int(off * sfx.SR)
        m = int(0.010 * sfx.SR)
        y[i:i + m] += rs.randn(m) * np.exp(-np.arange(m) / (0.0022 * sfx.SR)) * amp
    y += np.sin(2 * np.pi * 170 * k) * np.exp(-k * 90) * 0.5
    return np.convolve(y, np.ones(3) / 3, 'same') * 1.3


def audio(path):
    sr = sfx.SR
    buf = np.zeros(int((TOTAL + 1) * sr))
    rs = np.random.RandomState(11)
    # room tone: soft low noise + CRT hum
    t = np.arange(len(buf)) / sr
    noise = np.convolve(rs.randn(len(buf)), np.ones(40) / 40, 'same')
    buf += noise * 0.006 + np.sin(2 * np.pi * 59 * t) * 0.003 + np.sin(2 * np.pi * 15734 * t) * 0.0004
    def put(tt, s, g=1.0):
        i = int(tt * sr)
        if 0 <= i < len(buf):
            s = s[:len(buf) - i]
            buf[i:i + len(s)] += s * g
    for si in range(len(SC)):
        st = STARTS[si]
        put(st, sfx.synth('whoosh'), 0.13 if si else 0.0)
    put(OFF - 1.0, sfx.synth('whoosh'), 0.11)
    # soft exhale while the person sighs
    n = int(0.9 * sr)
    ex = np.convolve(rs.randn(n), np.ones(30) / 30, 'same') * np.sin(np.pi * np.arange(n) / n) ** 2
    put(OPEN + 1.0, ex, 0.10)
    if OPEN > 0:
        put(1.7, sfx.synth('whoosh'), 0.10)   # camera pan-in from the wide desk shot
    if DAY == '1':
        # slider moves click on their steps; pops where the green ok-labels land; ding on the reveal and the save
        for st_, ts in ((STARTS[2], (1.55, 1.95, 2.35)), (STARTS[3], (1.2, 1.7, 2.2)),
                        (STARTS[4], (0.6, 1.1, 1.6, 2.1, 2.6)), (STARTS[5], (1.0, 1.25, 1.5, 1.75))):
            for tt in ts:
                put(st_ + tt, dial(rs), 0.9)
        for st_, tt in ((STARTS[1], 2.5), (STARTS[1], 3.3), (STARTS[1], 4.0)):
            put(st_ + tt, sfx.synth('pop'), 0.30)
        put(STARTS[2] + 4.3, sfx.synth('pop'), 0.4)
        put(STARTS[3] + 3.6, sfx.synth('pop'), 0.4)
        put(STARTS[4] + 3.0, sfx.synth('pop'), 0.4)
        put(STARTS[6] + 0.2, sfx.synth('whoosh'), 0.11)
        put(STARTS[6] + 2.3, sfx.synth('ding'), 0.45)
        put(STARTS[7] + 2.6, sfx.synth('pop'), 0.4)
        put(STARTS[7] + 3.0, sfx.synth('ding'), 0.5)
    else:
        st = STARTS[1]
        for k in (1, 2, 3):     # shutter-speed dial clicks as it steps up
            put(st + 2.0 + k * 1.7 / 3.99, dial(rs), 1.0)
        put(st + 3.7, sfx.synth('pop'), 0.45)
        st = STARTS[3]
        put(st + 1.15, sfx.synth('click'), 0.9)
        put(st + 1.3, sfx.synth('ding'), 0.45)
        put(STARTS[4] + 3.7, sfx.synth('pop'), 0.4)
        put(STARTS[4] + 5.74, sfx.synth('pop'), 0.4)
        put(STARTS[6] + 4.3, sfx.synth('ding'), 0.5)
    buf = buf[:int(TOTAL * sr)]
    # fades
    n = int(0.5 * sr)
    buf[:n] *= np.linspace(0, 1, n)
    buf[-n:] *= np.linspace(1, 0, n)
    buf = buf / (np.max(np.abs(buf)) or 1) * 0.6
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((buf * 32767).astype(np.int16).tobytes())

if __name__ == '__main__':
    d = 'scenes'
    os.makedirs(d, exist_ok=True)
    os.makedirs('out', exist_ok=True)
    step = 93
    jobs = [(a, min(a + step, NFR), f'{d}/c{a:04d}.mp4') for a in range(0, NFR, step)]
    with mp.Pool(2) as p:
        rcs = p.map(render_range, jobs)
    assert all(r == 0 for r in rcs), rcs
    with open(f'{d}/list.txt', 'w') as f:
        for j in jobs:
            f.write(f"file '{os.path.abspath(j[2])}'\n")
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', f'{d}/list.txt', '-c', 'copy', f'{d}/video.mp4'])
    audio(f'{d}/sfx.wav')
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f'{d}/video.mp4', '-i', f'{d}/sfx.wav', '-map', '0:v', '-map', '1:a',
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', os.environ.get('CINE_OUT', 'out/day1_raw_vs_edit.mp4' if DAY == '1' else 'out/day2_shutter_speed_v4.mp4')])
    print('done')
