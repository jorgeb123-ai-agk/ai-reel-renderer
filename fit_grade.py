"""Fit the global edit that carries RAW toward FINAL, split into 3 steps: white balance, shadows (luminance curve),
saturation. Whatever is left (local edits) is a residual that is faded in with the last step so the video ends on the
real final photo. Writes grade_params.npz."""
import numpy as np, cv2
def load(p, s=(480, 320)):
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, s, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
raw, fin = load('assets/raw.jpg'), load('assets/final.jpg')
# 1 white balance: per-channel gains (green fixed) so the colour ratios of the means match the final
mr, mf = raw.reshape(-1, 3).mean(0), fin.reshape(-1, 3).mean(0)
q = mf / mr
G = q / q[1]
x = np.clip(raw * G, 0, 1)
# 2 shadows: monotone luminance curve (binned means, smoothed)
lum = lambda a: a @ np.array([0.299, 0.587, 0.114], np.float32)
l1, l2 = lum(x).ravel(), lum(fin).ravel()
knots = np.linspace(0, 1, 33)
ys = []
for k in knots:
    m = np.abs(l1 - k) < 0.02
    ys.append(l2[m].mean() if m.sum() > 200 else np.nan)
ys = np.array(ys)
ok = ~np.isnan(ys)
ys = np.interp(knots, knots[ok], ys[ok])
ys = np.maximum.accumulate(cv2.GaussianBlur(ys[None].astype(np.float32), (0, 0), 1.2)[0])
ys[0] = 0
def tone(a, s=1.0):
    l = lum(a)
    l2_ = l + s * (np.interp(l, knots, ys) - l)
    return np.clip(a * (l2_ / np.maximum(l, 1e-3))[..., None], 0, 1)
y = tone(x)
# 3 saturation scalar (least squares on chroma)
ly = lum(y)[..., None]; lf = lum(fin)[..., None]
cy, cf = (y - ly), (fin - lf)
Sat = float((cy * cf).sum() / (cy * cy).sum())
z = np.clip(ly + cy * Sat, 0, 1)
print('gains', G.round(3), 'sat', round(Sat, 3))
print('rmse raw', np.sqrt(np.mean((raw - fin) ** 2)).round(4), 'wb', np.sqrt(np.mean((x - fin) ** 2)).round(4),
      'tone', np.sqrt(np.mean((y - fin) ** 2)).round(4), 'sat', np.sqrt(np.mean((z - fin) ** 2)).round(4))
print('curve', np.round(ys[::4], 3))
np.savez('grade_params.npz', G=G, knots=knots, ys=ys, Sat=Sat)
