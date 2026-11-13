r"""Oracle test: how well can each output format keep drum hits when its gains are ideal?

    .venv\Scripts\python.exe oracle_hf.py <track folder> [<track folder> ...]

A track folder holds the four stems (drums.flac ... or .wav). For every format the drum stem is rebuilt from the
mixture with the ideal real gain per cell, |drums|^2 / sum of |stem|^2 (what a magnitude mask can do at best; the
mixture's phase is kept), and scored like hit_scan.py: SDR of the drums, SDR of their 6-16 kHz band, the level of
each hit, and the 6-16 kHz energy in the first 20 ms of each hit ("attack").

Formats:
  n2 rows      STFT 4096 / 1024, one gain per row of Needle 2's band grid (1, 2, 4, 16 bins)    = Needle 2 today
  4096 bins    STFT 4096 / 1024, one gain per bin: is it the rows or the window?
  n1 bins      STFT 2048 / 512, one gain per bin                                                 = Needle 1
  split        below 5.5 kHz as "n2 rows"; above it STFT 1024 / 256 with one gain per 4 bins (172 Hz, as wide as
               a Needle 2 row there) and per 5.8-ms frame: 4x the time resolution where hats live
  split 512    the same with STFT 512 / 128 above 5.5 kHz (2 bins = 172 Hz per gain)

No network is involved: this measures what the format allows, not what a small network learns in it.
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

from data import read_stems
from hit_scan import D, SR, onsets, window_db
from needle2 import bs_config, row_of_bin

CUT = 5512.5                                                    # Hz: the lower edge of Needle 2's 4-bin band


def stft(x, n, hop):
    return torch.stft(x, n, hop, window=torch.hann_window(n), center=True, return_complex=True)


def istft(X, n, hop, length):
    return torch.istft(X, n, hop, window=torch.hann_window(n), center=True, length=length)


def gains(P, rows=None):
    """P (S, C, F, T) powers of the stems -> ideal drum gain (C, F, T); with rows (F,), one gain per row."""
    num, den = P[D], P.sum(0)
    if rows is not None:
        idx = rows[None, :, None].expand_as(num)
        shape = (num.shape[0], int(rows.max()) + 1, num.shape[2])
        num = torch.zeros(shape).scatter_add_(1, idx, num)[:, rows]
        den = torch.zeros(shape).scatter_add_(1, idx, den)[:, rows]
    return num / (den + 1e-12)


def one(stems, mix, n, hop, rows=None, keep=None, unit=False):
    """Drums from the mixture with ideal gains in an STFT of size n. keep: bool (F,) of the bins this part covers."""
    X = stft(mix, n, hop)
    g = torch.ones(X.shape) if unit else gains(stft(stems.reshape(-1, stems.shape[-1]), n, hop).abs().pow(2).reshape(*stems.shape[:2], *X.shape[1:]), rows)
    if keep is not None:
        g = g * keep[None, :, None]
    return istft(g * X, n, hop, mix.shape[-1])


def n2_rows():
    r = row_of_bin(bs_config())
    r = torch.as_tensor(r).long()
    return torch.cat([r, r[-1:]])                              # the Nyquist bin joins the last row


def formats(stems, mix):
    r4096 = n2_rows()
    k4096 = torch.arange(2049) * SR / 4096
    out = {"n2 rows": one(stems, mix, 4096, 1024, r4096),
           "4096 bins": one(stems, mix, 4096, 1024),
           "n1 bins": one(stems, mix, 2048, 512)}
    low = one(stems, mix, 4096, 1024, r4096, keep=k4096 < CUT)
    for name, n, group in (("split", 1024, 4), ("split 512", 512, 2)):
        k = torch.arange(n // 2 + 1) * SR / n
        rows = torch.arange(n // 2 + 1) // group
        out[name] = low + one(stems, mix, n, n // 4, rows, keep=k >= CUT)
    # how exact is the split itself (gains of 1 in both parts)?
    whole = one(stems, mix, 4096, 1024, keep=k4096 < CUT, unit=True) + one(stems, mix, 1024, 256, keep=torch.arange(513) * SR / 1024 >= CUT, unit=True)
    err = 10 * torch.log10((mix - whole).pow(2).sum() / mix.pow(2).sum()).item()
    return out, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--seconds", type=float, default=0, help="only the first N seconds of each track (0 = all)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    hf = ss.butter(4, [6000, 16000], btype="band", fs=SR, output="sos")
    pooled, sdrs = {}, {}
    for t in a.tracks:
        d = Path(t)
        st = read_stems(d, "flac" if (d / "drums.flac").exists() else "wav", dtype="float32")
        if a.seconds:
            st = st[..., :int(a.seconds * SR)]
        stems = torch.from_numpy(np.ascontiguousarray(st))
        mix = stems.sum(0)
        ref = st[D].mean(0)
        hits = onsets(ref)
        hits = hits[(hits > 0) & (hits < len(ref) - SR // 10)]
        ref_hf = ss.sosfilt(hf, ref)
        t_lvl, t_att = window_db(ref, hits, int(0.05 * SR)), window_db(ref_hf, hits, int(0.02 * SR))
        live = t_att > t_att.max() - 30
        ests, err = formats(stems, mix)
        print(f"\n{d.name[:60]}: {len(hits)} hits, {live.sum()} with high-frequency content; the split alone leaves {err:.1f} dB of error")
        for name, e in ests.items():
            e = e.mean(0).numpy()
            e_hf = ss.sosfilt(hf, e)
            sdr = 10 * np.log10((ref ** 2).sum() / ((ref - e) ** 2).sum())
            sdr_hf = 10 * np.log10((ref_hf ** 2).sum() / ((ref_hf - e_hf) ** 2).sum())
            lvl = window_db(e, hits, int(0.05 * SR)) - t_lvl
            att = (window_db(e_hf, hits, int(0.02 * SR)) - t_att)[live]
            pooled.setdefault(name, [[], []])
            pooled[name][0].append(lvl)
            pooled[name][1].append(att)
            sdrs.setdefault(name, []).append((sdr, sdr_hf))
            print(f"  {name:>10}: drums SDR {sdr:5.2f} dB, 6-16 kHz SDR {sdr_hf:5.2f} dB | attack median {np.median(att):+.2f} dB, "
                  f">3 dB down {np.mean(att < -3) * 100:4.1f}%, >6 dB down {np.mean(att < -6) * 100:4.1f}% | level >10 dB down {np.mean(lvl < -10) * 100:.1f}%", flush=True)
    print(f"\nall {len(a.tracks)} tracks together")
    for name, (lv, at) in pooled.items():
        lv, at, s = np.concatenate(lv), np.concatenate(at), np.array(sdrs[name])
        print(f"  {name:>10}: drums SDR {s[:, 0].mean():5.2f} dB, 6-16 kHz SDR {s[:, 1].mean():5.2f} dB | attack median {np.median(at):+.2f} dB, "
              f">3 dB down {np.mean(at < -3) * 100:4.1f}%, >6 dB down {np.mean(at < -6) * 100:4.1f}%, >10 dB down {np.mean(at < -10) * 100:4.1f}% | "
              f"level >10 dB down {np.mean(lv < -10) * 100:.1f}%")


if __name__ == "__main__":
    main()
