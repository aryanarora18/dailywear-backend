#!/usr/bin/env python3
"""Score model predictions against human labels -> confusion matrices + recommendations.
Usage: python3 score.py [labels.jsonl]"""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "labels.jsonl"
rows = [json.loads(l) for l in open(path)]
print(f"labels: {len(rows)}\n")

verdicts = Counter(r["verdict"] for r in rows)
n = len(rows)
print("== box quality ==")
print(f"  real person:   {verdicts.get('label',0)} ({100*verdicts.get('label',0)//max(n,1)}%)")
print(f"  NOT a person:  {verdicts.get('not_person',0)} ({100*verdicts.get('not_person',0)//max(n,1)}%)  <- hallucinated boxes")
print(f"  can't tell:    {verdicts.get('cant_tell',0)} ({100*verdicts.get('cant_tell',0)//max(n,1)}%)  <- below resolution floor\n")

labeled = [r for r in rows if r["verdict"] == "label"]


def attr_report(attr, classes):
    conf = defaultdict(Counter)   # truth -> pred counts
    for r in labeled:
        conf[r["truth"][attr]][r["pred"][attr if attr != "bottoms" else "bottoms"]] += 1
    print(f"== {attr} ==")
    correct = sum(conf[c][c] for c in conf)
    total = sum(sum(v.values()) for v in conf.values())
    print(f"  overall accuracy: {correct}/{total} ({100*correct//max(total,1)}%)")
    print(f"  {'truth -> pred':24}", "  ".join(f"{c[:7]:>8}" for c in classes + ['unclear']))
    for t in classes:
        row = conf.get(t, Counter())
        tot = sum(row.values())
        if not tot:
            continue
        print(f"  {t:24}", "  ".join(f"{row.get(c,0):>8}" for c in classes + ['unclear']), f"  (n={tot})")
    # per-class precision/recall
    for c in classes:
        tp = conf.get(c, Counter()).get(c, 0)
        fn = sum(conf.get(c, Counter()).values()) - tp
        fp = sum(conf.get(t, Counter()).get(c, 0) for t in conf if t != c)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn)
        flag = "  <- WEAK, consider merging or hedging" if (prec < .75 or rec < .75) and tp + fn >= 8 else ""
        print(f"    {c:16} precision {prec:.2f}  recall {rec:.2f}{flag}")
    print()


attr_report("outer", ["jacket_or_coat", "hoodie", "long_sleeve", "short_sleeve", "sleeveless"])
attr_report("bottoms", ["long_pants", "shorts", "skirt_or_dress"])

umb_conf = Counter((r["truth"]["umbrella"], r["pred"]["umbrella"]) for r in labeled)
print("== umbrella ==")
print(f"  correct: {umb_conf.get((True,True),0)+umb_conf.get((False,False),0)}/{len(labeled)}"
      f"  false alarms: {umb_conf.get((False,True),0)}  missed: {umb_conf.get((True,False),0)}\n")

by_model = Counter(r["model"] for r in labeled)
if len(by_model) > 1:
    print("== by model ==")
    for m in by_model:
        sub = [r for r in labeled if r["model"] == m]
        ok = sum(1 for r in sub if r["pred"]["outer"] == r["truth"]["outer"] and r["pred"]["bottoms"] == r["truth"]["bottoms"])
        print(f"  {m}: both-attrs correct {ok}/{len(sub)}")
