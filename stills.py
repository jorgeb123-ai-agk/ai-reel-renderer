import sys, time, cv2, numpy as np
import cine
from PIL import Image
ts = [float(x) for x in sys.argv[2:]]
out = sys.argv[1]
tiles = []
t0 = time.time()
for t in ts:
    im = cine.render(int(t * 30))
    tiles.append(cv2.resize(im, (324, 576), interpolation=cv2.INTER_AREA))
print('per frame', (time.time() - t0) / len(ts))
cols = 5
rows = (len(tiles) + cols - 1) // cols
sheet = np.zeros((rows * 576, cols * 324, 3), np.uint8)
for i, tl in enumerate(tiles):
    sheet[(i // cols) * 576:(i // cols + 1) * 576, (i % cols) * 324:(i % cols + 1) * 324] = tl
Image.fromarray(sheet).save(out)
