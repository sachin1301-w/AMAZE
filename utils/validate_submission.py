#!/usr/bin/env python3
"""Validate matching_results.tsv / candidate_pairs.tsv against the challenge
rules.  Standard library only.

    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Prints PASS (exit 0) or a numbered list of problems (exit 1).
"""
import argparse
import csv
import os
import sys


def read_ids(path, col=0):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        head = next(r)
        for row in r:
            if row:
                out.append(row[col])
    return out, head


def load_table(path, id_col):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        head = next(r)
        if len(head) != 2 or head[0] != "source1_entity_id" or head[1] != id_col:
            return None, head, ["header must be "
                                f"'source1_entity_id\\t{id_col}', got {head}"]
        for i, row in enumerate(r, start=2):
            if not row:
                continue
            if len(row) == 1:
                row = [row[0], ""]
            if len(row) != 2:
                return None, head, [f"line {i}: expected 2 tab-separated "
                                    f"fields, got {len(row)}"]
            rows.append((row[0], row[1]))
    return rows, head, []


def check(rows, name, id_col, s1_ids, valid23, issues):
    seen = set()
    for e, lst in rows:
        if e in seen:
            issues.append(f"{name}: duplicate source1_entity_id {e}")
        seen.add(e)
        if lst == "":
            continue
        parts = lst.split(",")
        if len(set(parts)) != len(parts):
            issues.append(f"{name}: duplicate ids inside the list for {e}")
        for p in parts:
            if not (p.startswith("S2-") or p.startswith("S3-")):
                issues.append(f"{name}: {e} lists non Source-2/3 id {p}")
                break
            if p not in valid23:
                issues.append(f"{name}: {e} lists unknown id {p}")
                break
    missing = s1_ids - seen
    extra = seen - s1_ids
    if missing:
        issues.append(f"{name}: {len(missing)} Source-1 entities missing "
                      f"(e.g. {sorted(missing)[:3]})")
    if extra:
        issues.append(f"{name}: {len(extra)} unknown Source-1 entities "
                      f"(e.g. {sorted(extra)[:3]})")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    issues = []
    s1, _ = read_ids(os.path.join(a.test_dir, "test_source1.tsv"))
    s2, _ = read_ids(os.path.join(a.test_dir, "test_source2.tsv"))
    s3, _ = read_ids(os.path.join(a.test_dir, "test_source3.tsv"))
    s1_ids = set(s1)
    valid23 = set(s2) | set(s3)
    print(f"test set: {len(s1_ids):,} Source-1, {len(valid23):,} Source-2/3")

    mrows, _, e1 = load_table(a.matching, "matched_entity_ids")
    crows, _, e2 = load_table(a.candidate, "candidate_entity_ids")
    issues += e1 + e2
    if mrows is None or crows is None:
        for i, m in enumerate(issues, 1):
            print(f"{i}. {m}")
        return 1
    check(mrows, "matching_results", "matched_entity_ids", s1_ids, valid23, issues)
    check(crows, "candidate_pairs", "candidate_entity_ids", s1_ids, valid23, issues)

    cand = {e: set(l.split(",")) if l else set() for e, l in crows}
    nsub = 0
    for e, l in mrows:
        if not l:
            continue
        for p in l.split(","):
            if p not in cand.get(e, ()):
                nsub += 1
    if nsub:
        issues.append(f"WARNING: {nsub} matched ids never appeared as a "
                      f"candidate for the same entity")

    nm = sum(len(l.split(",")) for _, l in mrows if l)
    nc = sum(len(l.split(",")) for _, l in crows if l)
    print(f"matching_results: {nm:,} ids, {nm/len(mrows):.3f} per entity")
    print(f"candidate_pairs : {nc:,} ids, {nc/len(crows):.3f} per entity")
    print(f"reduction ratio : {nc/len(crows)/len(valid23):.3e} of the full "
          f"cross product")
    if issues:
        for i, m in enumerate(issues, 1):
            print(f"{i}. {m}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
