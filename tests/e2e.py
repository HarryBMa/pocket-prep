"""End-to-end check: import -> analyse -> stems -> export. Uses a throwaway data dir."""
import os, sys, time, json
os.environ.setdefault("POCKET_PREP_DATA", os.path.abspath("data/e2e"))
from pocket_prep import library, analysis, stems, db
from pocket_prep.export import usb
track = sys.argv[1]; out = sys.argv[2]
t0 = time.time()
ids = library.import_paths([track]) or [db.one("SELECT id FROM tracks WHERE path=?", (os.path.abspath(track),))["id"]]
tid = ids[0]; print("imported", tid, round(time.time()-t0, 1), "s")
a = analysis.analyse(tid); print("analysed", {k: a[k] for k in ("bpm", "key", "camelot", "energy")}, a["phrases"], round(time.time()-t0, 1), "s")
print("cues", library.cues(tid))
if "--stems" in sys.argv:
    stems.separate(tid, lambda f, m="": print(f"  stems {f:.0%} {m}", flush=True) if int(f*100) % 25 == 0 else None)
    print("stems", stems.stem_files(tid), round(time.time()-t0, 1), "s")
p = library.create_playlist("Test set"); library.add_to_playlist(p["id"], [tid])
print(json.dumps(usb.export(out), indent=1), round(time.time()-t0, 1), "s")
