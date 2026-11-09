# reel-renderer

Frame-by-frame, code-only renderer for faceless vertical sports-photography reels in a warm
"CRT monitor on a desk" look: a software-3D room with window-blind light, a retro monitor that
shows the creator's own photo and an amber UI, a typewriter headline on the wall, synthesized
sound effects, and a calm handheld camera. Output is 1080x1920, 30 fps, H.264 + AAC.

No game engine, no stock footage, no network at render time.

## Requirements
- Python 3.10+, `numpy`, `opencv-python`, `pillow`  (`pip install -r requirements.txt`)
- `ffmpeg` on the PATH
- A monospace bold font (DejaVu Sans Mono Bold is auto-detected; otherwise set `MONO_FONT=/path/to/font.ttf`)

## Quick start
```bash
pip install -r requirements.txt
cp /path/to/your/sharp_action_photo.jpg assets/photo.jpg     # landscape, ideally 3:2
python stills.py sheet.png 1.5 6 11 16 21 26 31 37           # contact sheet at those times (seconds)
python build.py                                              # full video -> out/day2_shutter_speed_v4.mp4
```
Environment variables:
- `CINE_END=21.93`  render only the first N seconds (trial cut)
- `CINE_OUT=out/name.mp4`  output file
- `PHOTO=/path/photo.jpg`  photo to use
- `MONO_FONT=/path/font.ttf`

A full 41 s render takes about 10 minutes on 2 cores (about 0.34 s per frame per core).
Master quality is CRF 10 (huge). For sharing, re-encode:
```bash
ffmpeg -i out/master.mp4 -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p -c:a copy -movflags +faststart share.mp4
```

## Files
| File | What it does |
|---|---|
| `cine.py` | The renderer: camera, textured quads, baked blind-light textures, CRT screen UI per scene, wall typewriter text, photo modes (`sharp`, `blur_all`, `blur_players`), intro silhouette, post-processing, `render(frame)` |
| `build.py` | Renders in 93-frame chunks on 2 processes, concatenates, builds the audio track, muxes |
| `sfx.py` | Numpy sound effects (whoosh, pop, click, ding) |
| `stills.py` | Renders a contact sheet of frames for quick visual checks |
| `assets/photo.jpg` | Your photo (not committed) |

## How a video is defined (Day 2: shutter speed)
- `SCENE_DURS` in `cine.py`: seconds per scene. `OFF = 3.6` is the intro before scene 1.
- `WLINES`: wall headline lines and the second (within the scene) each line starts typing.
- `screen_scene(...)`: draws the CRT screen for each scene index (-1 is the intro editor screen).
- `build.py` `audio()`: sound-effect events by time (dial clicks on shutter steps, pops, shutter click, ding). There are no typing sounds on purpose because a voice-over goes there.
- `KEYS` in `cine.py`: camera positions/targets over time.

## Making a new video
1. Copy the scene structure: new `SCENE_DURS`, `WLINES`, a branch per scene in `screen_scene`, and audio events.
2. Run `stills.py` and look at the contact sheet before any long render.
3. Trial cut with `CINE_END`, then one full render.
4. With voice clips: set each scene's duration to the clip length plus ~0.3 s and align `WLINES` start times and sound events to the words.

## Look settings (approved; do not regress)
Subtle blind-light stripes (wall `0.70 + 0.28*band`), film grain sigma 0.012, monitor scaled 1.10,
camera sway 0.6x, wall typing 17 characters/second, quiet room sound
(noise 0.006, hum 0.003), 1.5x supersampled 3D canvas and 1.5x supersampled screen texture.
