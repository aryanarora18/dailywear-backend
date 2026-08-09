#!/usr/bin/env python3
"""Golden regression: re-run the CURRENT model/prompt on labeled frames, score vs labels.
Run after any prompt or model change; compare to the last baseline printed.
Usage: GEMINI_API_KEY=... [DETECT_MODEL=...] python3 golden.py"""
import base64
import json
import os
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dailywear-backend"))
os.environ.setdefault("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
from main import gemini_people  # same prompt/schema as production  # noqa: E402


def iou(a, b):
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0, iy1, ix1 = max(ay0, by0), max(ax0, bx0), min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    ua = (ay1 - ay0) * (ax1 - ax0) + (by1 - by0) * (bx1 - bx0) - inter
    return inter / ua if ua else 0


labels = [json.loads(l) for l in open(os.path.join(HERE, "labels.jsonl")) if json.loads(l)["verdict"] == "label"]
by_frame = defaultdict(list)
for r in labels:
    stamp, cam_id = r["pid"].split("#")[0].split("/")
    by_frame[(stamp, cam_id)].append(r)

matched = correct_o = correct_b = missed = 0
for (stamp, cam_id), rows in by_frame.items():
    frame_path = os.path.join(HERE, "pool", stamp, f"{cam_id}.jpg")
    if not os.path.exists(frame_path):
        continue
    out = gemini_people(open(frame_path, "rb").read())
    preds = out.get("people", [])
    for r in rows:
        best = max(preds, key=lambda p: iou(p["box_2d"], r["pred"]["box_2d"]), default=None)
        if best is None or iou(best["box_2d"], r["pred"]["box_2d"]) < 0.4:
            missed += 1
            continue
        matched += 1
        correct_o += best["outer"] == r["truth"]["outer"]
        correct_b += best["bottoms"] == r["truth"]["bottoms"]

model = os.getenv("DETECT_MODEL", "gemini-flash-lite-latest")
print(f"GOLDEN [{model}] on {len(by_frame)} frames / {len(labels)} labeled people:")
print(f"  re-detected (IoU>=0.4): {matched}  missed: {missed}")
if matched:
    print(f"  outer accuracy:   {correct_o}/{matched} ({100*correct_o//matched}%)")
    print(f"  bottoms accuracy: {correct_b}/{matched} ({100*correct_b//matched}%)")
print("Compare against the previous run before shipping a prompt/model change.")
