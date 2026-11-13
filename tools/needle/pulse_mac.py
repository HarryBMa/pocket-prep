r"""Pulse on the Mac mini: copy training data, code and checkpoints between the PC and `brain` (ssh).

The PC's GPU and RAM belong to the Needle training, so Pulse trains on the Mac mini (Apple GPU, "mps").

    .venv\Scripts\python.exe pulse_mac.py push beat_teacher beat_teacher_ens beat_rave beat_loops_1t
    .venv\Scripts\python.exe pulse_mac.py verify beat_teacher beat_teacher_ens beat_rave beat_loops_1t
    .venv\Scripts\python.exe pulse_mac.py code                 pulse*.py -> brain:~/train/pulse
    .venv\Scripts\python.exe pulse_mac.py pull p5_w24_l6       brain:~/train/pulse/runs_pulse/<run> -> runs_pulse\<run>
    .venv\Scripts\python.exe pulse_mac.py fetch pulse_passfail  brain:~/train/data/<folder> -> F:\needle\<folder>: the files
                                                               the PC lacks (--overwrite: also those of another size)

push: F:\needle\<folder> -> brain:~/train/data/<folder>, only files that are missing there or have another
size (so it resumes), as one tar stream over ssh, then the same check as verify (names and sizes).
- --content compares every file's MD5 as well (a minute per 10 GB). Sizes alone are not enough after a copy
  that was cut off: the Mac's tar leaves the file it was writing at its full length, with zeros for the part
  that never arrived (two label files of the first copy on 12 Nov were like that). A set that was rewritten
  on the PC under the same file names needs --content too, or a new folder name.
- Windows' own tar.exe (bsdtar 3.8.9) crashes on file names outside the ANSI code page (Cyrillic track
  titles), so the tar stream is written by Python.
- Files are read without the Windows file cache: free RAM on the PC is the Needle training's memory-mapped
  data, and 15 GB of cached reads would push it out.
The index files keep their absolute Windows paths; pulse.local_path() maps them on the Mac (PULSE_DATA).
"""
import argparse
import ctypes
import hashlib
import mmap
import os
import subprocess
import sys
import tarfile
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(r"F:\needle")
HOST = "brain"
REMOTE_DATA = "~/train/data"
REMOTE_CODE = "~/train/pulse"
CODE = ["pulse.py", "pulse_rave.py", "pulse_raveform.py", "pulse_grid.py", "pulse_run_mac.sh", "pulse_passfail.py",
        "pulse_c/pulse_beat.cpp", "pulse_c/pulse_beat.h", "pulse_c/pulse_cli.cpp", "pulse_c/pulse_logits.cpp",
        "pulse_c/test_pulse_beat.cpp"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST]


def nfc(s):
    return unicodedata.normalize("NFC", s)


class Unbuffered:
    """A file read with FILE_FLAG_NO_BUFFERING: nothing goes through (or stays in) the Windows file cache."""
    CHUNK = 4 << 20                                             # a multiple of every sector size

    def __init__(self, path, size):
        from ctypes import wintypes as wt
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wt.HANDLE
        k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.LPVOID, wt.DWORD, wt.DWORD, wt.HANDLE]
        k32.ReadFile.restype = wt.BOOL
        k32.ReadFile.argtypes = [wt.HANDLE, wt.LPVOID, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.LPVOID]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        self.k32, self.wt = k32, wt
        self.h = k32.CreateFileW("\\\\?\\" + os.path.abspath(path), 0x80000000, 1 | 2, None, 3, 0x20000000, None)
        if self.h in (None, wt.HANDLE(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        self.buf = mmap.mmap(-1, self.CHUNK)                    # page-aligned, as unbuffered reads require
        self.view = ctypes.c_char.from_buffer(self.buf)
        self.left, self.pos, self.avail = size, 0, 0

    def read(self, n=-1):
        out, need = [], self.left + self.avail - self.pos if n is None or n < 0 else n
        while need > 0:
            if self.pos == self.avail:
                if self.left <= 0:
                    break
                got = self.wt.DWORD(0)
                if not self.k32.ReadFile(self.h, ctypes.addressof(self.view), self.CHUNK, ctypes.byref(got), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.avail, self.pos = min(got.value, self.left), 0
                self.left -= self.avail
                if self.avail == 0:
                    break
            k = min(need, self.avail - self.pos)
            out.append(self.buf[self.pos:self.pos + k])
            self.pos += k
            need -= k
        return b"".join(out)

    def close(self):
        self.k32.CloseHandle(self.h)
        del self.view
        self.buf.close()


def local_files(root: Path):
    """NFC name -> (relative posix path as it is on disk, size), for every file under root.
    Some track names are stored decomposed (o + combining accent): NTFS needs the exact form to open the
    file, the Mac's APFS takes either, so names are compared in NFC and sent as they are."""
    out = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            p = Path(dirpath) / n
            rel = p.relative_to(root).as_posix()
            out[nfc(rel)] = (rel, p.stat().st_size)
    return out


def remote_files(folder, md5=False):
    """NFC name -> size (or MD5) of every file in the folder on the Mac."""
    what = "md5 -r" if md5 else "stat -f '%z %N'"
    cmd = f"cd {REMOTE_DATA}/{folder} 2>/dev/null && find . -type f -exec {what} {{}} +"
    r = subprocess.run(SSH + [cmd], capture_output=True)
    out = {}
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        value, _, name = line.partition(" ")
        if name.startswith("./"):
            out[nfc(name[2:])] = value if md5 else int(value)
    return out


def local_md5(path, size):
    h = hashlib.md5()
    f = Unbuffered(path, size) if os.name == "nt" else open(path, "rb")
    try:
        while True:
            block = f.read(4 << 20)
            if not block:
                return h.hexdigest()
            h.update(block)
    finally:
        f.close()


def verify(folder, content=False):
    loc, rem = local_files(DATA / folder), remote_files(folder)
    missing = sorted(k for k in loc if k not in rem)
    wrong = sorted(k for k in loc if k in rem and rem[k] != loc[k][1])
    extra = sorted(k for k in rem if k not in loc)
    print(f"{folder}: {len(loc)} files, {sum(v[1] for v in loc.values()) / 2**30:.2f} GB on the PC; on the Mac {len(rem)} files, "
          f"{len(missing)} missing, {len(wrong)} with another size, {len(extra)} only on the Mac", flush=True)
    if content:
        t0, sums = time.time(), remote_files(folder, md5=True)
        same_size = [k for k in loc if k in rem and rem[k] == loc[k][1]]
        differ = sorted(k for k in same_size if sums.get(k) != local_md5(DATA / folder / loc[k][0], loc[k][1]))
        print(f"{folder}: {len(same_size)} files of the same size compared by MD5 in {time.time() - t0:.0f} s, "
              f"{len(differ)} with other content", flush=True)
        wrong += differ
    for k in (missing + wrong)[:5]:
        print("   ", k, flush=True)
    return missing + wrong, loc


def push(folder, content=False):
    todo, loc = verify(folder, content)
    if not todo:
        return True
    total = sum(loc[k][1] for k in todo)
    print(f"{folder}: sending {len(todo)} files, {total / 2**30:.2f} GB", flush=True)
    dest = f"{REMOTE_DATA}/{folder}"
    p = subprocess.Popen(SSH + [f"mkdir -p {dest} && tar -xf - -C {dest}"], stdin=subprocess.PIPE)
    t0, sent, last = time.time(), 0, time.time()
    with tarfile.open(fileobj=p.stdin, mode="w|", format=tarfile.PAX_FORMAT, bufsize=1 << 20) as tf:
        tf.copybufsize = 1 << 20
        for k in todo:
            src = DATA / folder / loc[k][0]
            info = tf.gettarinfo(str(src), arcname=loc[k][0])
            f = Unbuffered(src, info.size) if os.name == "nt" else open(src, "rb")
            try:
                tf.addfile(info, f)
            finally:
                f.close()
            sent += info.size
            if time.time() - last > 20:
                last = time.time()
                rate = sent / 2**20 / (last - t0)
                print(f"   {sent / 2**30:.2f} of {total / 2**30:.2f} GB, {rate:.0f} MB/s, "
                      f"{(total - sent) / 2**20 / max(rate, 1e-6) / 60:.0f} min left", flush=True)
    p.stdin.close()
    rc = p.wait()
    print(f"{folder}: sent in {(time.time() - t0) / 60:.1f} min, remote tar exit {rc}", flush=True)
    return not verify(folder, content)[0]


def code(to=REMOTE_CODE):
    files = [HERE / n for n in CODE if (HERE / n).exists()]
    p = subprocess.Popen(SSH + [f"mkdir -p {to} && tar -xf - -C {to}"], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
        for f in files:
            info = tf.gettarinfo(str(f), arcname=f.relative_to(HERE).as_posix())
            info.mode = 0o755 if f.suffix == ".sh" else 0o644
            data = f.read_bytes().replace(b"\r\n", b"\n")       # the shell script must not carry CRLF
            info.size = len(data)
            import io
            tf.addfile(info, io.BytesIO(data))
    p.stdin.close()
    print("code:", ", ".join(f.name for f in files), "->", to, "exit", p.wait(), flush=True)


def fetch(folder, overwrite=False):
    """The other way: files of ~/train/data/<folder> that the PC lacks -> F:\\needle\\<folder>. A file the PC has at
    another size stays as it is unless overwrite: it may be the newer one (an executable rebuilt on the PC after
    the folder was pushed, a result.json with the audio run added), and the Mac's copy would replace it."""
    loc, rem = local_files(DATA / folder) if (DATA / folder).exists() else {}, remote_files(folder)
    missing = sorted(k for k in rem if k not in loc)
    differ = sorted(k for k in rem if k in loc and loc[k][1] != rem[k])
    todo = missing + (differ if overwrite else [])
    print(f"{folder}: {len(rem)} files on the Mac, {len(missing)} the PC lacks, {len(differ)} it has at another size"
          + (" (taken from the Mac)" if overwrite and differ else " (left as they are; --overwrite takes the Mac's)" if differ else ""), flush=True)
    for k in differ[:20]:
        print("    another size:", k, flush=True)
    if not todo:
        return True
    names = "\n".join(todo).encode("utf-8")
    r = subprocess.run(SSH + [f"cd {REMOTE_DATA}/{folder} && COPYFILE_DISABLE=1 tar -cf - -T -"], input=names, capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode("utf-8", "replace")[-500:], flush=True)
        return False
    import io
    (DATA / folder).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(DATA / folder, filter="data")
    for k in todo[:20]:
        print("   ", k, flush=True)
    return True


def pull(run, code_dir=REMOTE_CODE):
    dest = HERE / "runs_pulse" / run
    dest.mkdir(parents=True, exist_ok=True)
    # COPYFILE_DISABLE: without it the Mac's tar adds a "._name" file of extended attributes next to every file
    r = subprocess.run(SSH + [f"cd {code_dir}/runs_pulse/{run} && COPYFILE_DISABLE=1 tar -cf - ."], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"no such run on the Mac: {run}")
    import io
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
        tf.extractall(dest, filter="data")
    print(f"{run}: {', '.join(sorted(names))} -> {dest}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "verify", "code", "pull", "fetch"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--content", action="store_true", help="push / verify: compare every file's MD5, not only its size")
    ap.add_argument("--overwrite", action="store_true", help="fetch: also take the files the PC has at another size")
    ap.add_argument("--to", help="code / pull: another folder under ~/train than pulse (e.g. pulse_x for an experiment, so a "
                                 "queued run keeps the code it started with)")
    a = ap.parse_intermixed_args()                              # --content may stand before the folder names
    ok = True
    for n in a.names if a.cmd != "code" else [None]:
        if a.cmd == "push":
            ok &= push(n, a.content)
        elif a.cmd == "verify":
            ok &= not verify(n, a.content)[0]
        elif a.cmd == "pull":
            pull(n, f"~/train/{a.to}" if a.to else REMOTE_CODE)
        elif a.cmd == "fetch":
            ok &= fetch(n, a.overwrite)
        else:
            code(f"~/train/{a.to}" if a.to else REMOTE_CODE)
    print("ALL OK" if ok else "INCOMPLETE", flush=True)
    sys.exit(0 if ok else 1)
