#!/usr/bin/env python3
"""Pull the eval pool from GCS and build manifest.json for the labeler.
Usage: python3 pull.py"""
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
env = dict(os.environ, CLOUDSDK_PYTHON=os.path.expanduser(
    "~/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12"))

subprocess.run([GCLOUD, "storage", "rsync", "-r", "gs://dailywear-state/eval_pool",
                os.path.join(HERE, "pool")], env=env, check=True)

samples = []
for jp in sorted(glob.glob(os.path.join(HERE, "pool", "*", "*.json"))):
    meta = json.load(open(jp))
    frame = jp[:-5] + ".jpg"
    if not os.path.exists(frame):
        continue
    samples.append({"key": f"{meta['stamp']}/{meta['cam_id']}",
                    "frame": os.path.relpath(frame, HERE),
                    "cam": meta["cam"], "stamp": meta["stamp"],
                    "model": meta.get("model", "?"), "people": meta["people"]})
json.dump(samples, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
n_people = sum(len(s["people"]) for s in samples)
print(f"manifest: {len(samples)} frames · {n_people} predicted people to judge")
print("next: python3 label_server.py  → open http://localhost:8400/label.html")
