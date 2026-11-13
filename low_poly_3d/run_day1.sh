#!/usr/bin/env bash
# Day 1 (5 s) low-poly 3D reel: scene -> frames -> sound effects -> captions -> mp4 with sound.
# Usage: bash run_day1.sh            (full: 120 frames, about 45 min on 2 CPU cores)
#        FRAMES="1 12 1" bash run_day1.sh   (quick test: frames 1..12)
set -euo pipefail
cd "$(dirname "$0")"
export REEL_ROOT="$PWD"
export SFX_KIT="${SFX_KIT:-$PWD/vendor/sfx-kit}"
SFX_COMMIT=92780e7ab0cdfb557497a988089c47f46e8c7412   # vetted 2026-11-13

if [ ! -d "$SFX_KIT/.git" ]; then
  mkdir -p vendor
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/wyzowl/sfx-kit "$SFX_KIT"
fi
HEAD_NOW=$(git -C "$SFX_KIT" rev-parse HEAD)
if [ "$HEAD_NOW" != "$SFX_COMMIT" ]; then
  echo "WARNING: sfx-kit is at $HEAD_NOW, vetted commit is $SFX_COMMIT. Review what changed before using it."
fi

pip install --break-system-packages -r requirements.txt >/dev/null 2>&1 || pip install -r requirements.txt

# frames: resumable (existing frames are skipped). args: first last step samples width height
read -r A B C <<< "${FRAMES:-1 120 1}"
python3 blend_scene.py -- "$A" "$B" "$C" 24 540 960

python3 sfx_mix.py       # writes sfx_raw.wav, sfx_final.wav (-14 LUFS), cues_day1.json
python3 captions.py      # upscales to 1080x1920, adds captions, encodes, muxes sound
echo "done: $REEL_ROOT/day1_reel.mp4"
