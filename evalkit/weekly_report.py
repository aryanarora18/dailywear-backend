#!/usr/bin/env python3
"""Weekly usefulness report: calibration bias, sweep health, and user feedback.
Usage: python3 weekly_report.py"""
import json
import os
import subprocess
from collections import defaultdict

GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
env = dict(os.environ, CLOUDSDK_PYTHON=os.path.expanduser(
    "~/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12"))


def cat(obj):
    r = subprocess.run([GCLOUD, "storage", "cat", f"gs://dailywear-state/{obj}"],
                       env=env, capture_output=True, text=True)
    return r.stdout


print("=" * 60)
print("DAILYWEAR WEEKLY REPORT")
print("=" * 60)

cal = [json.loads(l) for l in cat("calibration.jsonl").splitlines() if l.strip()]
by_day = defaultdict(list)
for r in cal:
    by_day[r["ts"][:10]].append(r)
print(f"\n== calibration: implied vs official temp ({len(cal)} sweeps) ==")
for day in sorted(by_day):
    rows = [r for r in by_day[day] if r["n"] >= 25]
    ns = [r["n"] for r in by_day[day]]
    if rows:
        bias = sum(r["implied"] - r["official"] for r in rows) / len(rows)
        print(f"  {day}: {len(by_day[day])} sweeps · median n={sorted(ns)[len(ns)//2]} · bias {bias:+.1f}°")
    else:
        print(f"  {day}: {len(by_day[day])} sweeps · all below n=25 (thin day)")
solid = [r for r in cal if r["n"] >= 25]
if solid:
    overall = sum(r["implied"] - r["official"] for r in solid) / len(solid)
    print(f"  OVERALL BIAS: {overall:+.1f}° -> suggested mapping offset: {-overall:+.0f}°")

thin = sum(1 for r in cal if r["n"] < 25)
print(f"\n== sweep health ==\n  {len(cal)} sweeps · {thin} thin (n<25) · "
      f"{100*(len(cal)-thin)//max(len(cal),1)}% healthy")

fb = [json.loads(l) for l in cat("feedback.jsonl").splitlines() if l.strip()]
print(f"\n== user feedback ({len(fb)}) ==")
for r in fb[-25:]:
    print(f"  [{r['ts'][:16]}] {r['text'][:120]}")
if not fb:
    print("  none yet — share the site!")
