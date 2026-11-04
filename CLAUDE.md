# Pocket Prep: notes for Claude

- Only stop processes by exact PID (e.g. `taskkill /PID 1234`, `kill 1234`), never by name or
  image (`taskkill /IM ...`, `pkill`, `killall`): other programs on this machine (e.g. the local
  AI setup's Python processes, other app instances) must never be hit.
- Commit after every change (one commit per logical change). See HANDOVER.md for layout and gaps.
- Prefer existing libraries over hand-written code (ClubCraft, rekordcrate, wavesurfer.js);
  check licences and that everything works offline.
