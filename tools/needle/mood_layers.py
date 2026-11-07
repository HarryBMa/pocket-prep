"""Experiment: how deep into htdemucs does the mood embedding need to reach? (for a smaller model file)

Like mood_embed.py, but keeps the pooled output (mean and standard deviation) of every encoder layer of both branches
and of the cross-transformer separately, so the head can be tested on layers 0..k only. Also prints the parameter count
of each part, which is what the model file size follows.
Usage (Needle venv): python mood_layers.py <list.tsv> <out.npz>
"""
import sys
import numpy as np
import torch
from demucs.pretrained import get_model
from mood_embed import load, SR


def main():
    model = get_model("htdemucs").models[0].cuda().eval()
    parts = {f"freq_enc{i}": m for i, m in enumerate(model.encoder)}
    parts.update({f"time_enc{i}": m for i, m in enumerate(model.tencoder)})
    parts["transformer"] = model.crosstransformer
    parts["freq_dec"] = model.decoder
    parts["time_dec"] = model.tdecoder
    for k, m in parts.items():
        print(f"{k:12s} {sum(p.numel() for p in m.parameters()) / 1e6:6.2f} M parameters")
    print("total", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    grabbed = {}
    for i, m in enumerate(model.encoder):
        m.register_forward_hook(lambda mod, inp, out, i=i: grabbed.__setitem__(f"f{i}", out))
    for i, m in enumerate(model.tencoder):
        m.register_forward_hook(lambda mod, inp, out, i=i: grabbed.__setitem__(f"t{i}", out))
    model.crosstransformer.register_forward_hook(lambda mod, inp, out: grabbed.__setitem__("x", out))
    labels, files, vecs = [], [], []
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8") if "\t" in l]
    seg = int(7.8 * SR)
    for n, (label, path) in enumerate(rows):
        try:
            wav = load(path, 120)
            feats = []
            for i in range(0, wav.shape[1] - seg + 1, seg):
                x = torch.from_numpy(wav[:, i:i + seg])[None].cuda()
                ref = x.mean(0)
                x = (x - ref.mean()) / (ref.std() + 1e-8)
                grabbed.clear()
                with torch.no_grad():
                    model(x)
                row = []
                for k in range(4):                                    # layers 0..3: both branches, mean and std
                    f, t = grabbed[f"f{k}"], grabbed[f"t{k}"]
                    row += [f.mean((2, 3)), f.std((2, 3)), t.mean(2), t.std(2)]
                fq, tm = grabbed["x"]
                row += [fq.mean((2, 3)), fq.std((2, 3)), tm.mean(2), tm.std(2)]
                feats.append(torch.cat(row, 1)[0].cpu().numpy())
            vecs.append(np.mean(feats, axis=0))
            labels.append(label)
            files.append(path.replace("\\", "/").rsplit("/", 1)[-1])
        except Exception as e:
            print("skip", path, e, file=sys.stderr)
        if n % 20 == 0:
            print(n, len(rows), flush=True)
    np.savez(sys.argv[2], x=np.stack(vecs), labels=np.array(labels), files=np.array(files))
    print("saved", np.stack(vecs).shape)


if __name__ == "__main__":
    main()
