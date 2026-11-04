#!/usr/bin/env bash
# End-to-end run against a throwaway library: import, analyse, stems (first
# track only), export, then check the card layout. Works on macOS and in Git Bash.
#
#   tests/e2e.sh <export folder> <audio file>...
#
# Needs: a built app (cargo build --release) and models/htdemucs.onnx.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$1"; shift
DATA="$(mktemp -d)"
BIN=target/release/pocket-prep
[ -x "$BIN.exe" ] && BIN="$BIN.exe"

POCKET_PREP_ANY_TARGET=1 POCKET_PREP_DATA="$DATA" "$BIN" --headless > "$DATA/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do grep -q "running at" "$DATA/server.log" && break; sleep 0.2; done
URL=$(grep -o 'http://127.0.0.1:[0-9]*/' "$DATA/server.log")
echo "server: $URL  data: $DATA"

api() { curl -sf -H 'Content-Type: application/json' "$@"; }
json_paths=$(printf '"%s",' "$@"); json_paths="[${json_paths%,}]"
# Windows paths need forward slashes inside JSON strings.
json_paths=${json_paths//\\//}

t0=$(date +%s)
echo "import: $(api -d "{\"paths\": $json_paths}" "${URL}api/import")"

wait_jobs() {
  while api "${URL}api/jobs" | grep -qE '"status":"(queued|running)"'; do sleep 1; done
  if api "${URL}api/jobs" | grep -q '"status":"failed"'; then
    echo "FAILED JOB:"; api "${URL}api/jobs"; exit 1
  fi
}
wait_jobs
echo "analysis done in $(( $(date +%s) - t0 )) s"
api "${URL}api/tracks" | grep -oE '"(title|bpm|musical_key|camelot|energy)":[^,]*' | paste - - - - - | sed 's/^/  /'

t0=$(date +%s)
api -d '{"ids": [1]}' "${URL}api/stems" > /dev/null
wait_jobs
echo "stems for track 1 in $(( $(date +%s) - t0 )) s"

rm -rf "$OUT"
JOB=$(api -d "{\"root\": \"${OUT//\\//}\", \"include_stems\": true}" "${URL}api/export" | grep -o '[0-9]*')
wait_jobs
echo "export: $(api "${URL}api/jobs" | grep -o '"result":{[^}]*')"

for f in PIONEER/rekordbox/export.pdb POCKETDECK/stems.json POCKETDECK/export.json; do
  [ -s "$OUT/$f" ] || { echo "missing $f"; exit 1; }
done
ls "$OUT"/POCKETDECK/STEMS/*.flac > /dev/null
# Audio must be copied byte for byte.
for src in "$@"; do
  dst=$(find "$OUT/Contents" -name "$(basename "$src")" | head -1)
  cmp -s "$src" "$dst" || { echo "not byte-identical: $src"; exit 1; }
done
echo "OK: card layout complete, audio byte-identical"
