"""What a checkpoint of train.py holds, on one line: which weights, step, validation SDR.

    .venv\\Scripts\\python ckinfo.py runs\\n2_d\\best.pt      ->  ema0.999 120000 5.612

"raw" = the trained weights; "ema<decay>" = an averaged copy (train.py --ema). Checkpoints of runs without --ema
are "raw"."""
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ck.get("weights", "raw"), ck.get("step", 0), f"{ck.get('val_avg', float('nan')):.3f}")
