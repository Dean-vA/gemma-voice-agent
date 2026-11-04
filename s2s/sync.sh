#!/usr/bin/env sh
# Copy the speech-server code from the Reachy (Ruxithid) project into s2s/app.
#
# The Reachy project is the source of truth: these files are what its backend.py runs on
# Windows, and this stack runs the same code in containers. Rerun after changing them there.
#
#   sh s2s/sync.sh "/c/Users/deanv/Dropbox/Personal Dev Projects/Reachy-dnd-character"
#
# Only code is copied. Voiceprints, the roster, table logs, profiles and campaign notes stay
# in the Reachy project: this repo is public.
set -eu
src="${1:?usage: sh s2s/sync.sh <path to Reachy-dnd-character>}"
dst="$(dirname "$0")/app"
mkdir -p "$dst/speaker_id"
for f in backend.py gestures.py memory.py heard.py table_log.py dm_only.py pc_audio.py; do
  cp "$src/$f" "$dst/$f"
done
cp "$src/speaker_id/server.py" "$dst/speaker_id/server.py"
echo "copied into $dst"
