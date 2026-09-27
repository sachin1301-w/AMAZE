"""Build the final submission zip.

    python package.py --team <team_name>

Produces <team_name>_submission.zip with the layout the challenge asks for:

    output/{matching_results.tsv,candidate_pairs.tsv}
    code/business_entity_resolution/{src,README.md,requirements.txt}
    Documentation_template.md
"""
import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # code/business_entity_resolution
ROOT = os.path.abspath(os.path.join(PKG, "..", ".."))

SRC_SKIP = {"__pycache__"}
SRC_KEEP_EXT = {".py", ".json", ".md", ".txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="submission")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, f"{a.team}_submission.zip")
    need = [os.path.join(ROOT, "output", "matching_results.tsv"),
            os.path.join(ROOT, "output", "candidate_pairs.tsv"),
            os.path.join(PKG, "README.md"),
            os.path.join(PKG, "requirements.txt"),
            os.path.join(ROOT, "Documentation_template.md")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit("missing:\n  " + "\n  ".join(missing))

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(ROOT, "output", f), f"output/{f}")
            n += 1
        for f in ("README.md", "requirements.txt"):
            z.write(os.path.join(PKG, f), f"code/business_entity_resolution/{f}")
            n += 1
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in SRC_SKIP]
            for f in sorted(files):
                if os.path.splitext(f)[1] not in SRC_KEEP_EXT:
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, HERE).replace(os.sep, "/")
                z.write(p, f"code/business_entity_resolution/src/{rel}")
                n += 1
        z.write(os.path.join(ROOT, "Documentation_template.md"),
                "Documentation_template.md")
        n += 1
    print(f"wrote {out}  ({n} files, {os.path.getsize(out)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
