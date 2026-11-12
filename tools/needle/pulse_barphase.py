r"""Downbeats as the deck uses them: is the bar line on the right beat? Raveform test set, p2_w24.
For each track: the share of human downbeats that the estimate puts on the right beat (not just near a beat),
for bar-fit windows of 8 / 16 / 32 bars and Beat This!. 'Bar right' = more than 90 % of the track's bars right."""
import sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pulse import Pulse, f_measure, peaks, predict_track    # noqa: E402
from pulse_rave import Rave, bar_fit                         # noqa: E402

def bar_share(ref_d, est_d, tol=0.07):
    if len(ref_d) == 0 or len(est_d) == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(est_d - r)) <= tol for r in ref_d]))

te = Rave("test")
ck = torch.load(HERE / "runs_pulse" / "p2_w24" / "last.pt", map_location="cpu", weights_only=False)
m = Pulse(ck["width"]); m.load_state_dict(ck["model"]); m = m.cuda().eval()
from beat_this.inference import Spect2Frames
from beat_this.model.postprocessor import Postprocessor
s2f = Spect2Frames(checkpoint_path=r"F:\needle\beat_this\final0.ckpt", device="cuda", float16=False)
post = Postprocessor(type="minimal")
res = {k: [] for k in ("beat_this", "bars8", "bars16", "bars32", "whole")}
for t in te.ids:
    b, d = te.lab[t]
    mel = np.asarray(te.mels[t], dtype=np.float32)
    with torch.inference_mode():
        bl, dl = s2f(torch.from_numpy(mel.T).cuda())
        _, dd = post(bl, dl)
        y = predict_track(m, mel, "cuda")
    eb = peaks(y[0])
    res["beat_this"].append(bar_share(d, np.asarray(dd)))
    for n in (8, 16, 32):
        res[f"bars{n}"].append(bar_share(d, bar_fit(eb, y[1], bars=n)))
    res["whole"].append(bar_share(d, bar_fit(eb, y[1], bars=4000)))
print(f"{len(te.ids)} test tracks: share of human downbeats hit | tracks with >90 % of bars right")
for k, v in res.items():
    v = np.array(v)
    print(f"{k:10s} {v.mean():.3f} | {np.mean(v > 0.9) * 100:.0f}%")
