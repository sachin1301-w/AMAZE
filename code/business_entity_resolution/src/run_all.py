"""End-to-end driver: data -> blocking -> matching -> output/.

    python run_all.py                 # everything
    python run_all.py --from fit      # resume at a stage

Stages are idempotent: each writes its artefacts under work/ and is skipped if
they already exist (delete them to force a rerun).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "dataset"))
OUT = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
PY = sys.executable

STAGES = ["prep", "dfstats", "dftab", "maps", "train_pairs", "fit",
          "test_pairs", "submit"]


def sh(args, **kw):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"          # blocking hashes must be reproducible
    env["PYTHONUTF8"] = "1"
    env.setdefault("BER_WORK", WORK)
    env.setdefault("BER_DATA", DATA)
    env.update(kw.pop("env", {}))
    print("\n$ " + " ".join(args), flush=True)
    t = time.time()
    r = subprocess.run([PY] + args, cwd=HERE, env=env)
    if r.returncode:
        sys.exit(f"stage failed: {args}")
    print(f"  ({time.time()-t:.0f}s)", flush=True)


def have(*names):
    return all(os.path.exists(os.path.join(WORK, n)) for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="prep", choices=STAGES)
    ap.add_argument("--keepfrac", type=float, default=0.15,
                    help="fraction of training entities kept for the matcher")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    i0 = STAGES.index(a.start)

    def want(name):
        return STAGES.index(name) >= i0

    if want("prep"):
        for sp in ("train", "test"):
            if a.force or not have(f"{sp}_s1.parquet", f"{sp}_s3.parquet"):
                sh(["prep.py", sp])
    if want("dfstats"):
        for sp in ("train", "test"):
            if a.force or not have(f"{sp}_df.pkl"):
                sh(["dfstats.py", sp])
    if want("dftab"):
        for sp in ("train", "test"):
            if a.force or not have(f"{sp}_dftab.npz"):
                sh(["mkdftab.py", sp])
    if want("maps"):
        if a.force or not os.path.exists(os.path.join(HERE, "learned_maps.json")):
            sh(["learn_maps.py"])
    if want("train_pairs"):
        sh(["runpipe.py", "train", "--tag", "full",
            "--keepfrac", str(a.keepfrac)])
    if want("fit"):
        sh(["fit.py", "--tag", "full"])
    if want("test_pairs"):
        sh(["runpipe.py", "test", "--tag", "pred",
            "--model", os.path.join(WORK, "model_full_s1.lgb"),
            "--model2", os.path.join(WORK, "model_full_s2.lgb")])
    if want("submit"):
        sh(["submit.py", "--tag", "pred", "--outdir", OUT,
            "--decide", os.path.join(WORK, "model_full_decide.json")])
    print("\ndone ->", OUT)


if __name__ == "__main__":
    main()
