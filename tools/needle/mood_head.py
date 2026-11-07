"""Needle Tags, part 2: the small head on top of the embedding (mood_embed.py), tested honestly.

A head is a multinomial logistic regression: standardise every number, multiply by a weight matrix, softmax.
With a few hundred tracks and 1,500+ numbers per track it would learn the training tracks by heart, so it is
held back by an L2 penalty (lambda). Lambda is chosen on the training tracks only (inner 3-fold
cross-validation), and the accuracy is counted on tracks the head never saw (outer stratified k-fold).

Usage (the Needle venv):
  python mood_head.py --luke emb_luke.npz --jam emb_jam.npz [--keep keep.txt] [--only deep,melodic,tech_house,peaktime]
                      [--features all|embedding|stems] [--save head.json]
Labels come from the npz ("?" = unlabelled, ignored). --keep limits the files used (one name per line).
"""
import argparse, json, collections
import numpy as np
import torch

MOODS = ["deep", "melodic", "tech_house", "peaktime", "hard_techno", "dnb", "pop", "downtempo"]
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def stratified_folds(y, k, seed=1):
    rng, fold = np.random.RandomState(seed), np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def group_folds(y, groups, k, seed=1):
    """Folds that keep every group (an artist) in one fold, as even per mood as a greedy assignment can make them."""
    rng, fold = np.random.RandomState(seed), np.zeros(len(y), int)
    names = list(dict.fromkeys(groups))
    rng.shuffle(names)
    size = collections.Counter(groups)
    load = np.zeros((k, y.max() + 1))
    for g in sorted(names, key=lambda g: -size[g]):
        idx = [i for i, q in enumerate(groups) if q == g]
        add = np.bincount(y[idx], minlength=y.max() + 1)
        f = int(np.argmin([((load[j] + add) ** 2).sum() for j in range(k)]))
        load[f] += add
        fold[idx] = f
    return fold


def train_head(x, y, n_classes, lam, steps=200):
    """Multinomial logistic regression with L2, by L-BFGS. Returns (mean, std, W, b)."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd).float()
    yt = torch.from_numpy(y).long()
    W = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")
    counts = np.bincount(y, minlength=n_classes).clip(1)
    wt = torch.from_numpy(len(y) / (n_classes * counts)).float()          # every mood counts the same

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ W + b, yt, weight=wt) + lam * (W ** 2).sum() / x.shape[1]
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, W.detach().numpy(), b.detach().numpy()


def predict(head, x):
    mu, sd, W, b = head
    return ((x - mu) / sd @ W + b).argmax(1)


def pick_lambda(x, y, n_classes):
    """Best lambda by inner 3-fold cross-validation on the training tracks only."""
    fold, best, score = stratified_folds(y, 3, seed=2), LAMBDAS[0], -1
    for lam in LAMBDAS:
        ok = 0
        for f in range(3):
            h = train_head(x[fold != f], y[fold != f], n_classes, lam)
            ok += (predict(h, x[fold == f]) == y[fold == f]).sum()
        if ok > score:
            best, score = lam, ok
    return best


def cross_validate(x, y, n_classes, k, groups=None):
    fold = group_folds(y, groups, k) if groups is not None else stratified_folds(y, k)
    conf, per, lams = collections.Counter(), [], []
    for f in range(k):
        tr, te = fold != f, fold == f
        lam = pick_lambda(x[tr], y[tr], n_classes)
        lams.append(lam)
        p = predict(train_head(x[tr], y[tr], n_classes, lam), x[te])
        per.append((p == y[te]).mean())
        for a, b in zip(y[te], p):
            conf[(a, b)] += 1
    return per, conf, lams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luke")
    ap.add_argument("--jam")
    ap.add_argument("--keep")
    ap.add_argument("--only", default="")
    ap.add_argument("--features", default="all", choices=["all", "embedding", "stems"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--groups", help="TSV of 'file<TAB>group' (artist): a group is never split between training and test")
    ap.add_argument("--save")
    a = ap.parse_args()
    xs, ls, fs = [], [], []
    for p in (a.luke, a.jam):
        if p:
            d = np.load(p)
            xs.append(d["x"]); ls += list(d["labels"]); fs += list(d["files"])
    x, labels, files = np.concatenate(xs), np.array(ls), np.array(fs)
    keep = {l.strip() for l in open(a.keep, encoding="utf-8")} if a.keep else None
    only = a.only.split(",") if a.only else MOODS
    m = np.array([(l in only) and (keep is None or f in keep) for l, f in zip(labels, files)])
    x, labels, files = x[m], labels[m], files[m]
    if a.features == "embedding":
        x = x[:, :-8]
    elif a.features == "stems":
        x = x[:, -8:]
    ids = [mo for mo in MOODS if (labels == mo).sum() >= a.folds]
    sel = np.isin(labels, ids)
    x, labels, files = x[sel], labels[sel], files[sel]
    groups = None
    if a.groups:
        gm = dict(l.rstrip("\n").split("\t") for l in open(a.groups, encoding="utf-8") if "\t" in l)
        groups = [gm.get(f, "single:" + f) for f in files]
        groups = [("single:" + f) if g.endswith(":?") else g for g, f in zip(groups, files)]   # unknown artist = alone
    y = np.array([ids.index(l) for l in labels])
    print("tracks per mood:", ", ".join(f"{i} {(labels == i).sum()}" for i in ids), "| numbers per track:", x.shape[1])
    per, conf, lams = cross_validate(x, y, len(ids), a.folds, groups)
    n = sum(conf.values()); acc = sum(v for (p, q), v in conf.items() if p == q) / n
    print(f"\n{a.folds}-fold {'artist-grouped ' if groups else ''}cross-validation (tracks the head never saw): {100 * acc:.1f} % correct; folds {', '.join(f'{100 * p:.0f}' for p in per)} %; lambdas {lams}")
    print("labelled \\ predicted".ljust(22) + "".join(i[:9].rjust(10) for i in ids) + "    recall")
    for p in range(len(ids)):
        row = [conf[(p, q)] for q in range(len(ids))]
        print(ids[p].ljust(22) + "".join(str(v).rjust(10) for v in row) + f"    {100 * row[p] / max(sum(row), 1):.0f} %")
    if a.save:
        lam = float(np.median(lams))
        mu, sd, W, b = train_head(x, y, len(ids), lam)
        json.dump({"moods": ids, "lambda": lam, "mean": mu.tolist(), "std": sd.tolist(), "W": W.tolist(), "b": b.tolist()}, open(a.save, "w"))
        print("saved", a.save)


if __name__ == "__main__":
    main()
