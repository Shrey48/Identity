#!/usr/bin/env python3
"""
Builds fixed_question_set.json from the ANONYMOUS run's output files.

It copies each run's `problem_statement` exactly as it was asked (question text,
option A, option B, A/B order, correct answer, metadata), so every later
condition - and a re-run of anonymous - asks the identical 300 questions.

USAGE
    python3 make_fixed_question_set.py /path/to/folder/with/anonymous/json/files
Writes fixed_question_set.json next to this script (where dataset_loaders.py
looks for it). Refuses to write unless the set is exactly:
    MMLU 100 (25 per subject) + MoralChoice 100 (50 low / 50 high) + Moral Machine 100
"""
import os, sys, json
from collections import Counter

def dataset_key(ds):
    d = str(ds).lower()
    for k in ("mmlu", "moralchoice", "moralmachine"):
        if k in d:
            return k
    return d

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    folder = os.path.abspath(sys.argv[1])
    found = {}
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.endswith("_anonymous.json"):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                ps = json.load(f).get("problem_statement")
            if not ps or "id" not in ps:
                continue
            if ps["id"] in found and found[ps["id"]] != ps:
                sys.exit(f"ERROR: question {ps['id']} appears twice with different content.")
            found[ps["id"]] = ps

    questions = sorted(found.values(), key=lambda q: (dataset_key(q["dataset"]), q["id"]))
    per_ds = Counter(dataset_key(q["dataset"]) for q in questions)
    per_mmlu = Counter(q["category"] for q in questions if dataset_key(q["dataset"]) == "mmlu")
    per_mc = Counter(q["category"] for q in questions if dataset_key(q["dataset"]) == "moralchoice")
    print("Questions found per dataset:", dict(per_ds))
    print("MMLU per subject:", dict(per_mmlu))
    print("MoralChoice per ambiguity:", dict(per_mc))

    problems = []
    for k in ("mmlu", "moralchoice", "moralmachine"):
        if per_ds.get(k, 0) != 100: problems.append(f"{k}: {per_ds.get(k, 0)} questions (expected 100)")
    for subj, n in per_mmlu.items():
        if n != 25: problems.append(f"MMLU {subj}: {n} questions (expected 25)")
    if len(per_mmlu) != 4: problems.append(f"MMLU has {len(per_mmlu)} subjects (expected 4)")
    for amb, n in per_mc.items():
        if n != 50: problems.append(f"MoralChoice {amb}: {n} questions (expected 50)")
    if len(questions) != 300: problems.append(f"total {len(questions)} questions (expected 300)")
    if problems:
        print("NOT WRITTEN - the set is not the expected 300:")
        for p in problems: print("  -", p)
        sys.exit(1)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_question_set.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"total": len(questions),
                   "per_dataset": dict(per_ds), "questions": questions}, f, indent=2, ensure_ascii=False)
    print(f"Written: {out}  ({len(questions)} questions)")

if __name__ == "__main__":
    main()
