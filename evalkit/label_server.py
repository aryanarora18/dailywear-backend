#!/usr/bin/env python3
"""Serves the labeler UI and appends submitted labels to labels.jsonl.
Usage: python3 label_server.py  → open http://localhost:8400/label.html"""
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


class H(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            row = json.loads(self.rfile.read(n))
            with open("labels.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def done_keys():
    try:
        return {json.loads(l)["pid"] for l in open("labels.jsonl")}
    except FileNotFoundError:
        return set()


print(f"labeled so far: {len(done_keys())}")
print("labeler: http://localhost:8400/label.html")
HTTPServer(("127.0.0.1", 8400), H).serve_forever()
