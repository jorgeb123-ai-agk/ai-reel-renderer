"""Synthesized sound effects (numpy only)."""
import numpy as np

SR = 44100


def synth(kind):
    rs = np.random.RandomState(3)
    if kind == 'whoosh':
        n = int(0.45 * SR)
        x = rs.randn(n)
        k = np.linspace(0, 1, n)
        env = np.sin(np.pi * k) ** 2
        s1 = np.convolve(x, np.ones(70) / 70, 'same')
        s2 = np.convolve(x, np.ones(9) / 9, 'same')
        return (s1 * (1 - k) + s2 * k) * env * 0.9
    if kind == 'pop':
        n = int(0.14 * SR)
        k = np.arange(n) / SR
        f = 650 * np.exp(-k * 20) + 160
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 28) * 0.55
    if kind == 'click':
        n = int(0.14 * SR)
        y = np.zeros(n)
        for off, amp in ((0, 1.0), (int(0.055 * SR), 0.7)):
            m = int(0.014 * SR)
            y[off:off + m] += rs.randn(m) * np.exp(-np.arange(m) / (0.0035 * SR)) * amp
        return np.convolve(y, np.ones(5) / 5, 'same') * 1.6
    if kind == 'ding':
        n = int(0.5 * SR)
        k = np.arange(n) / SR
        return (np.sin(2 * np.pi * 988 * k) + 0.4 * np.sin(2 * np.pi * 1976 * k)) * np.exp(-k * 7) * 0.35
    raise ValueError(kind)
