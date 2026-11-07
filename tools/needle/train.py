"""Train Needle (needle.py) on MUSDB18-HQ and, optionally, teacher stems made by distill.py.

    # base 16, 32-frame chunks, MUSDB only
    python train.py --musdb E:\\needle-data\\musdb18hq --base 16 --frames 32 --name b16_f32

    # add teacher stems from your own music (half of the stems drawn from them)
    python train.py --musdb ... --teacher E:\\needle-data\\teacher --teacher-share 0.5 --name b16_f32_kd

    # add real stems from import_stems.py (a quarter of the stems drawn from them)
    python train.py --musdb ... --stems E:\\needle-data\\stems\\ni --stems E:\\needle-data\\stems\\mshoxx --stems-share 0.25 --name ...

    # continue a run that was stopped (time limit, reboot, Ctrl+C)
    python train.py --musdb ... --name b16_f32 --resume

Writes runs/<name>/:
  best.pt   best validation SDR so far: {"model", "config", ...}, what export_vela.py --checkpoint reads
  last.pt   everything needed to resume (model, optimiser, schedule, step)
  log.csv   training losses every --log-every steps;  val.csv  validation SDR per source
  TensorBoard events:  tensorboard --logdir runs

One training step, in words:
  1. take a batch of random crops of the four stems (int16, from RAM) and copy it to the GPU;
  2. augment every stem (gain, stereo swap, polarity) and add them up: that is the mixture;
  3. STFT the mixture and the stems; the network sees log1p |mixture| and predicts four masks;
  4. loss = L1(masked mixture magnitude, stem magnitude) + 0.1 x L1(waveform after iSTFT, stem).
"""
import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from chunked import nsdr, separate_chunked, torch_predictor
from data import (Prefetcher, StemBank, augment, hours, keep_awake, load_excerpts, load_musdb, load_teacher,
                  musdb_split)
from needle import SOURCES, Needle, NeedleConfig, apply_masks, count_macs, istft, network_input

HERE = Path(__file__).parent


# ── STFT on training crops ───────────────────────────────────────────────────

def crop_length(cfg: NeedleConfig):
    """Samples in one training crop: exactly `frames` STFT frames with center=False."""
    return (cfg.frames - 1) * cfg.hop + cfg.n_fft          # 17,920 for 32 frames


def stft_frames(x, cfg: NeedleConfig):
    """(..., L) -> complex STFT (..., n_fft//2+1, frames).

    center=False means no padding: every frame lies entirely inside the crop, so it holds real
    music on both sides, exactly like a chunk cut from the device's continuous STFT stream.
    (needle.spectrogram uses center=True on whole tracks; away from the track start the frames
    are the same.)"""
    lead = x.shape[:-1]
    spec = torch.stft(x.reshape(-1, x.shape[-1]), cfg.n_fft, cfg.hop, center=False, return_complex=True,
                      window=torch.hann_window(cfg.n_fft, device=x.device))
    return spec.reshape(*lead, *spec.shape[-2:])


def check_alignment(cfg: NeedleConfig, device):
    """Masks of 1 must give back the input: iSTFT(STFT(crop)) == the middle of the crop.
    This proves the waveform target below is cut from the right samples."""
    x = torch.randn(2, 2, crop_length(cfg), device=device)
    y = istft(stft_frames(x, cfg)[:, None], cfg, (cfg.frames - 1) * cfg.hop)[:, 0]
    h = cfg.n_fft // 2
    err = (y - x[..., h:h + y.shape[-1]]).abs().max().item()
    assert err < 1e-4, f"STFT/iSTFT alignment is off: {err}"


def compute_loss(model, stems, cfg: NeedleConfig, wave_weight, amp_dtype, spec_norm, mr_weight=0.0):
    """stems (B, S, C, L) float on the GPU. Returns (total, spectral L1, waveform L1)."""
    mix = stems.sum(1)                                            # (B, C, L)
    mix_spec = stft_frames(mix, cfg)                              # (B, C, F, T)
    with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        masks = model(network_input(mix_spec, cfg))               # (B, S*C, bins, T), sigmoid
    masks = masks.float()

    # Spectral loss. apply_masks normalises the masks to sum to 1, exactly as on the device.
    # Magnitudes are divided by sqrt(sum(window^2)) so a spectrogram value and a waveform sample
    # are on the same scale; then "0.1 x waveform loss" really means a tenth of the weight.
    est_mag = apply_masks(masks, mix_spec.abs(), cfg)             # (B, S, C, F, T)
    tgt_mag = stft_frames(stems, cfg).abs()
    spec_l1 = (est_mag - tgt_mag).abs().mean() / spec_norm

    # Waveform loss: masked complex mixture -> iSTFT -> compare with the middle of the crop.
    est_wav = istft(apply_masks(masks, mix_spec, cfg), cfg, (cfg.frames - 1) * cfg.hop)
    h = cfg.n_fft // 2
    tgt_wav = stems[..., h:h + est_wav.shape[-1]]
    wave_l1 = (est_wav - tgt_wav).abs().mean()
    loss = spec_l1 + wave_weight * wave_l1
    if mr_weight:
        loss = loss + mr_weight * multires_loss(est_wav, tgt_wav)
    return loss, spec_l1, wave_l1


MR_WINDOWS = (2048, 1024, 512, 256)


def multires_loss(est, tgt):
    """Multi-resolution complex-spectrogram L1 (as BS-RoFormer and Moises-Light train): the
    separated waveforms are compared at several STFT sizes, so short windows check transients
    (drum hits) and long windows check pitch. Each resolution is scaled like the main spectral
    loss (divided by sqrt(sum window^2)), and the resolutions are averaged."""
    e, t = est.reshape(-1, est.shape[-1]), tgt.reshape(-1, tgt.shape[-1])
    total = 0.0
    for n in MR_WINDOWS:
        w = torch.hann_window(n, device=est.device)
        se = torch.stft(e, n, n // 4, window=w, center=False, return_complex=True)
        st = torch.stft(t, n, n // 4, window=w, center=False, return_complex=True)
        total = total + (torch.view_as_real(se) - torch.view_as_real(st)).abs().mean() / w.pow(2).sum().sqrt()
    return total / len(MR_WINDOWS)


def change_speed(stems, out_len, max_change, p, gen=None):
    """Speed/pitch augmentation (Demucs' "repitch", done by resampling): with probability p an
    example is played up to `max_change` faster or slower (0.06 = about +-1 semitone and +-6 %
    tempo), all its stems by the same factor so the mixture still matches. stems (B, S, C, L_in)
    with L_in >= out_len * (1 + max_change); returns (B, S, C, out_len). Linear interpolation:
    the tiny high-frequency loss it causes is the same in the mixture and in the targets."""
    b = stems.shape[0]
    f = 1 + (torch.rand(b, device=stems.device) * 2 - 1) * max_change
    f = torch.where(torch.rand(b, device=stems.device) < p, f, torch.ones_like(f))
    pos = torch.arange(out_len, device=stems.device)[None] * f[:, None]          # (B, out_len)
    i0 = pos.floor().long().clamp_(max=stems.shape[-1] - 2)
    frac = (pos - i0)[:, None, None]
    idx = i0[:, None, None].expand(b, *stems.shape[1:3], out_len)
    a0 = stems.gather(-1, idx)
    a1 = stems.gather(-1, idx + 1)
    return a0 + (a1 - a0) * frac


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, tracks, cfg: NeedleConfig, overlap, device):
    """Separate every validation track with device-style chunking (float) and return the
    per-track SDR (tracks, sources). The mixture is the sum of the stems."""
    predict = torch_predictor(model)
    scores = []
    for _, stems in tracks:
        ref = torch.from_numpy(stems).float() / 32768             # reference stays on the CPU
        est = separate_chunked(predict, ref.sum(0).to(device), cfg, overlap)
        scores.append(nsdr(ref, est))
    model.train()
    torch.cuda.empty_cache()                                      # give validation's memory back before training
    return torch.stack(scores).float()


# ── Checkpoints ──────────────────────────────────────────────────────────────

def save_model(path, model, cfg, **extra):
    """The format export_vela.py --checkpoint reads: {"model": state_dict, "config": asdict(cfg)}."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "config": asdict(cfg), **extra}, tmp)
    tmp.replace(path)                                             # never leave a half-written file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--musdb", required=True, help="MUSDB18-HQ root (with train/ and test/)")
    ap.add_argument("--teacher", action="append", default=[], help="distill.py cache folder (repeatable)")
    ap.add_argument("--teacher-share", type=float, default=0.5, help="fraction of stems drawn from teacher tracks")
    ap.add_argument("--teacher-seconds", type=int, default=0,
                    help="keep only the busiest N seconds of each teacher track (0 = whole tracks); saves RAM on long DJ tracks")
    ap.add_argument("--stems", action="append", default=[], help="import_stems.py folder of real stems (repeatable)")
    ap.add_argument("--stems-share", type=float, default=0.25, help="fraction of stems drawn from --stems tracks")
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    # model (only these may change; the layer types are fixed for the NPU)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--mult", type=int, nargs="+", default=list(NeedleConfig.mult))
    ap.add_argument("--frames", type=int, default=32)
    # optimisation
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr", type=float, default=0.01, help="final lr as a fraction of --lr")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--mr-weight", type=float, default=0.0,
                    help="weight of the multi-resolution complex STFT loss on the separated waveforms (0 = off)")
    ap.add_argument("--p-speed", type=float, default=0.0, help="probability of speed/pitch augmentation (0 = off)")
    ap.add_argument("--speed-change", type=float, default=0.06, help="largest speed change, 0.06 = +-6 %%")
    ap.add_argument("--wave-weight", type=float, default=0.1)
    # augmentation
    ap.add_argument("--remix", type=float, default=0.5, help="probability that a crop mixes stems of different songs")
    ap.add_argument("--gain-db", type=float, default=6.0)
    ap.add_argument("--p-swap", type=float, default=0.5)
    ap.add_argument("--p-flip", type=float, default=0.5)
    # bookkeeping
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30, help="stop after this many validations without a new best (validation SDR is noisy: 8 stopped runs far too early)")
    ap.add_argument("--overlap", type=int, default=8, help="chunk overlap in frames (validation)")
    ap.add_argument("--max-hours", type=float, default=11.5, help="save and stop after this long (resume later)")
    ap.add_argument("--gpu-mem", type=float, default=0.8,
                    help="cap on PyTorch's share of GPU memory; on Windows going over spills into system RAM (~100x slower)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    keep_awake()
    device = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True                         # pick the fastest conv kernels once
    torch.cuda.set_per_process_memory_fraction(a.gpu_mem)         # fail loudly instead of spilling silently
    run = Path(a.runs) / a.name
    run.mkdir(parents=True, exist_ok=True)

    cfg = NeedleConfig(base=a.base, mult=tuple(a.mult), frames=a.frames)
    cfg.check()
    check_alignment(cfg, device)
    model = Needle(cfg).to(device)
    info = count_macs(cfg)
    print(f"Needle base {cfg.base} mult {cfg.mult} frames {cfg.frames}: {info['params']:,} parameters, "
          f"{info['gmac_per_audio_s']:.2f} GMAC per second of audio")

    # Weight decay on conv weights only: shrinking BatchNorm scales or biases towards 0 doesn't
    # regularise anything useful, it just fights the normalisation.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, fused=True)

    def lr_factor(step):                                          # linear warm-up, then cosine decay
        if step < a.warmup:
            return (step + 1) / a.warmup
        p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
        return a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16")  # bf16 has fp32's range: no scaling needed

    start, best, bad = 0, -float("inf"), 0
    if a.resume:
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        assert NeedleConfig(**ck["config"]) == cfg, "the checkpoint was trained with another config"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["step"], ck["best"], ck["bad"]
        print(f"resumed at step {start:,}, best validation SDR {best:.2f} dB")

    # Data
    t = time.time()
    split = musdb_split(a.musdb)
    train = load_musdb(a.musdb, "train", split["train"])
    valid = load_musdb(a.musdb, "train", split["valid"])
    teacher = [tr for d in a.teacher for tr in (load_excerpts(d, a.teacher_seconds) if a.teacher_seconds else load_teacher(d))]       # teacher stems (distill.py)
    real = [tr for d in a.stems for tr in load_teacher(d)]            # real stems (import_stems.py), same layout
    share_t = a.teacher_share if teacher else 0.0
    share_r = a.stems_share if real else 0.0
    assert share_t + share_r < 1, "MUSDB needs a share too"
    groups = [(train, 1.0 - share_t - share_r)] + ([(teacher, share_t)] if teacher else []) + ([(real, share_r)] if real else [])
    print(f"data in RAM after {time.time() - t:.0f} s: MUSDB train {len(train)} tracks ({hours(train):.1f} h), "
          f"valid {len(valid)} ({hours(valid):.1f} h), teacher {len(teacher)} ({hours(teacher):.1f} h), "
          f"real stems {len(real)} ({hours(real):.1f} h); shares MUSDB {1 - share_t - share_r:.2f}, "
          f"teacher {share_t:.2f}, real {share_r:.2f}")
    crop = crop_length(cfg)
    bank = StemBank(groups, math.ceil(crop * (1 + a.speed_change)) + 2 if a.p_speed else crop)
    batches = Prefetcher(bank, a.batch, a.remix, seed=a.seed + start)
    spec_norm = torch.hann_window(cfg.n_fft).pow(2).sum().sqrt().item()

    # Logs. On resume, rows logged after the checkpoint's step are dropped (they'll be redone),
    # and the hours column continues from the checkpoint, so it counts training time only.
    new = not a.resume or not (run / "log.csv").exists()
    hours_offset = 0.0
    if not new:
        for fname in ("log.csv", "val.csv"):
            rows = list(csv.reader(open(run / fname, newline="")))
            rows = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= start]
            with open(run / fname, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
            if fname == "log.csv" and len(rows) > 1:
                hours_offset = float(rows[-1][1])
    tb = SummaryWriter(str(run), purge_step=start if a.resume else None)
    log_f = open(run / "log.csv", "w" if new else "a", newline="")
    val_f = open(run / "val.csv", "w" if new else "a", newline="")
    log, val = csv.writer(log_f), csv.writer(val_f)
    if new:
        log.writerow(["step", "hours", "loss", "spec_l1", "wave_l1", "grad_norm", "lr", "steps_per_s"])
        val.writerow(["step", "hours", *SOURCES, "avg", "best"])

    model.train()
    t_wall = t_log = time.time()                                  # this session, for --max-hours
    t_start = t_wall - hours_offset * 3600                        # cumulative, for the logs
    acc = torch.zeros(4, device=device)
    step = start
    while step < a.steps:
        stems = batches.next().to(device, non_blocking=True).float().div_(32768)
        if a.p_speed:
            stems = change_speed(stems, crop, a.speed_change, a.p_speed)
        stems = augment(stems, a.gain_db, a.p_swap, a.p_flip)
        loss, spec_l1, wave_l1 = compute_loss(model, stems, cfg, a.wave_weight, amp_dtype, spec_norm, a.mr_weight)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        acc += torch.stack([loss.detach(), spec_l1.detach(), wave_l1.detach(), grad_norm.detach()])

        if step % a.log_every == 0:                               # one GPU sync per log line, not per step
            m = (acc / a.log_every).tolist()
            acc.zero_()
            if not math.isfinite(m[0]):
                raise SystemExit(f"loss is {m[0]} at step {step}: stopping (check the data and the learning rate)")
            sps = a.log_every / (time.time() - t_log)
            t_log = time.time()
            lr = sched.get_last_lr()[0]
            h = (time.time() - t_start) / 3600
            log.writerow([step, f"{h:.4f}", *(f"{v:.6f}" for v in m), f"{lr:.3e}", f"{sps:.2f}"])
            log_f.flush()
            for k, v in zip(("loss", "spec_l1", "wave_l1", "grad_norm"), m):
                tb.add_scalar(f"train/{k}", v, step)
            tb.add_scalar("train/lr", lr, step)
            eta = (a.steps - step) / sps / 3600
            print(f"step {step:>7,}  loss {m[0]:.4f} (spec {m[1]:.4f}, wave {m[2]:.4f})  |g| {m[3]:.2f}  "
                  f"lr {lr:.2e}  {sps:.1f} steps/s  {h:.2f} h, {eta:.1f} h left", flush=True)

        done = step >= a.steps or (time.time() - t_wall) / 3600 > a.max_hours
        if step % a.val_every == 0 or done:
            sdr = validate(model, valid, cfg, a.overlap, device)  # (tracks, sources)
            med = sdr.median(0).values                            # median over tracks, as in SiSEC/museval
            avg = med.mean().item()
            improved = avg > best
            best, bad = (avg, 0) if improved else (best, bad + 1)
            h = (time.time() - t_start) / 3600
            val.writerow([step, f"{h:.4f}", *(f"{v:.3f}" for v in med.tolist()), f"{avg:.3f}", f"{best:.3f}"])
            val_f.flush()
            for s, v in zip(SOURCES, med.tolist()):
                tb.add_scalar(f"val_sdr/{s}", v, step)
            tb.add_scalar("val_sdr/avg", avg, step)
            print(f"validation at step {step:,}: " + ", ".join(f"{s} {v:.2f}" for s, v in zip(SOURCES, med.tolist()))
                  + f"  -> avg {avg:.2f} dB {'(new best)' if improved else f'(best {best:.2f})'}", flush=True)
            extra = {"step": step, "val_sdr": dict(zip(SOURCES, med.tolist())), "val_avg": avg}
            if improved:
                save_model(run / "best.pt", model, cfg, **extra)
            save_model(run / "last.pt", model, cfg, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                       scaler=scaler.state_dict(), best=best, bad=bad, **extra)
            if bad >= a.patience:
                print(f"no improvement in {a.patience} validations: stopping early")
                break
        if done:
            break

    print(f"finished at step {step:,} after {(time.time() - t_start) / 3600:.2f} h; best validation SDR {best:.2f} dB")
    tb.close()
    log_f.close()
    val_f.close()


if __name__ == "__main__":
    main()
