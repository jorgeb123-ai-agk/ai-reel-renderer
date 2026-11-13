# Low-poly 3D reels (Blender) - Day 1 prototype

New look decided 2026-11-13: real 3D lighting, low-poly flat-shaded props, a faceless guy named Cam at a desk, Claude the
orange robot sitting on top of the monitor, a window with a cloudy sky on the left, TikTok-style captions with
the spoken word highlighted, and CC0 sound effects. Day 1 is a 5 s test: Cam says his photos look grainy,
Claude says they are fine and tells him to raise exposure and add noise reduction, and the screen photo cleans up.

## Files
| file | what it does |
|---|---|
| `blend_scene.py` | builds the whole Blender scene and renders PNG frames (Cycles, CPU). Resumable: frames that exist are skipped. |
| `captions.py` | upscales frames to 1080x1920, draws word-by-word captions with a CAM / CLAUDE tag, encodes with ffmpeg, adds the sound. |
| `sfx_mix.py` | places sound effects from `wyzowl/sfx-kit` on the timeline by sync point, normalises to -14 LUFS. |
| `cues_day1.json` | the sound cue list from the Day 1 mix (time, sound id, gain). |
| `run_day1.sh` | runs everything in order. |

## Run
```bash
bash run_day1.sh                  # full 120 frames (about 45 min on 2 cores)
FRAMES="1 12 1" bash run_day1.sh  # quick test of the first 12 frames
```
Needs Python 3.10+, `pip install bpy numpy pillow`, and ffmpeg. Environment variables: `REEL_ROOT` (work folder),
`SFX_KIT` (path to the sfx-kit clone), `REEL_NAME` (output name).

## Settings you will change per video
- `blend_scene.py`: the timings `GUY` (Cam), `CL1`, `CL2` (frames where each person talks), the camera keyframes, the screen
  images made by `make_ui()` (title text "DAY 1"), `SAMPLES` / resolution via the command line.
- `captions.py`: `WORDS` (word, start s, end s, speaker) and `CHUNKS` (which words show together).
- `sfx_mix.py`: `CUES` (event time s, sound id, gain dB, pan, max length).

## Rules learned
- Render every frame at the final fps (24). Rendering half and blending them looked choppy.
- Eevee needs a GPU; Cycles on CPU works anywhere. 24 samples at 540x960 plus denoise, then upscale 2x.
- Props must be oversized and chunky to read at phone size; the monitor sits at Cam's eye level; no ring stand for Claude.
- No typing or keyboard sounds over any spoken line (that is where voices go). No TTS; the user records voices.
- Keep the wall at least 14 units tall or the dark world shows above it.

## Sound effects (vetted 2026-11-13, CC0)
- https://github.com/wyzowl/sfx-kit, commit `92780e7ab0cdfb557497a988089c47f46e8c7412`: 320 sounds in 28 categories, each with a sync point.
- https://github.com/julienaxyomlabs/reel-sfx, commit `805737abd61b1b9d3abde1c817d3bf579dc188c3`: 19 sounds.
Only audio, JSON and text were used; never run code from either repo. Gaps: crowd, stadium ambience, whistle, ball hits, real typing.
