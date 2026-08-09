"""DailyWear backend: cameras -> Gemini -> aggregate -> spectrum -> warm copy -> one JSON.

Env: GEMINI_API_KEY (required) · GEMINI_MODEL · DATA_DIR · SWEEP_TOKEN (protects POST /api/sweep)
Endpoints: GET /api/today · POST /api/feedback · POST /api/sweep · GET /api/frame/{cam_id} · GET /healthz
"""
import base64
import concurrent.futures as cf
import json
import os
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NYC = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(NYC)
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.getenv("DATA_DIR", os.path.join(HERE, "data"))
os.makedirs(os.path.join(DATA, "history"), exist_ok=True)
os.makedirs(os.path.join(DATA, "frames"), exist_ok=True)
CAMS = json.load(open(os.path.join(DATA, "cams.json")))
UA = {"User-Agent": "dailywear/1.0 (dailywear.nyc)"}
_lock = threading.Lock()

app = FastAPI(title="dailywear")
app.add_middleware(CORSMiddleware, allow_origins=["https://dailywear.nyc", "http://dailywear.nyc",
                                                  "http://localhost:8200", "http://localhost:8300"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------- vision ----------

PROMPT = (
    "This is a low-resolution NYC street camera frame. Find every clearly visible PEDESTRIAN "
    "(sidewalks/crosswalks; ignore people in vehicles, reflections, tiny distant figures — only "
    "people taller than about 1/10 of the frame). For each: box_2d [ymin,xmin,ymax,xmax] 0-1000; "
    "outer (jacket_or_coat, hoodie, long_sleeve, short_sleeve, sleeveless, unclear); bottoms "
    "(shorts, long_pants, skirt_or_dress, unclear); top_color (dominant color of their top garment); "
    "umbrella; confidence 0-1. Use 'unclear' freely "
    "when you cannot really tell. Also scene_ok: true only if lighting/visibility would let a human "
    "verify your answers. Strictly JSON."
)
SCHEMA = {"type": "OBJECT", "properties": {
    "scene_ok": {"type": "BOOLEAN"},
    "people": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "outer": {"type": "STRING", "enum": ["jacket_or_coat", "hoodie", "long_sleeve", "short_sleeve", "sleeveless", "unclear"]},
        "bottoms": {"type": "STRING", "enum": ["shorts", "long_pants", "skirt_or_dress", "unclear"]},
        "top_color": {"type": "STRING", "enum": ["black", "white", "gray", "blue", "denim", "red", "green",
                                                  "yellow", "orange", "pink", "purple", "brown", "beige",
                                                  "patterned", "unclear"]},
        "umbrella": {"type": "BOOLEAN"}, "confidence": {"type": "NUMBER"}},
        "required": ["box_2d", "outer", "bottoms", "top_color", "umbrella", "confidence"]}}},
    "required": ["scene_ok", "people"]}


_key_i = 0
def _keys() -> list:
    ks = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    return [k.strip() for k in ks.split(",") if k.strip()]


def gemini_people(frame: bytes) -> dict:
    global _key_i
    keys = _keys()
    model = os.getenv("DETECT_MODEL", "gemini-flash-lite-latest")
    body = {"contents": [{"parts": [{"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(frame).decode()}}]}],
            "generationConfig": {"response_mime_type": "application/json", "response_schema": SCHEMA}}
    for attempt in range(2 * len(keys) + 1):
        key = keys[_key_i % len(keys)]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        r = requests.post(url, json=body, timeout=60)
        if r.status_code == 429:
            _key_i += 1              # rotate to the other project's quota
            if attempt >= len(keys):
                time.sleep(20)       # both throttled: brief backoff
            continue
        r.raise_for_status()
        return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    raise RuntimeError("rate limited on all keys")


MIN_H = 70   # min person box height (of 1000)
MIN_CONF = 0.6


def sweep_cam(cam: dict) -> Optional[dict]:
    try:
        frame = requests.get(cam["url"], headers=UA, timeout=20).content
        if len(frame) < 4000:
            print(f"[cam] {cam['name']}: placeholder frame", flush=True)
            return None
        out = gemini_people(frame)
        if not out.get("scene_ok"):
            print(f"[cam] {cam['name']}: scene not ok", flush=True)
            return None
        people = [p for p in out["people"]
                  if p.get("confidence", 0) >= MIN_CONF and len(p.get("box_2d", [])) == 4
                  and (p["box_2d"][2] - p["box_2d"][0]) >= MIN_H]
        open(os.path.join(DATA, "frames", f"{cam['id']}.jpg"), "wb").write(frame)
        return {"cam": cam["name"], "cam_id": cam["id"], "people": people}
    except Exception as e:
        print(f"[cam] {cam['name']}: {str(e)[:80]}", flush=True)
        return None


# ---------- weather (NWS, free) ----------

def nws() -> dict:
    try:
        h = requests.get("https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly", headers=UA, timeout=15).json()
        d = requests.get("https://api.weather.gov/gridpoints/OKX/33,37/forecast", headers=UA, timeout=15).json()
        hp = h["properties"]["periods"]
        now = hp[0]
        tonight = next((p for p in hp if datetime.fromisoformat(p["startTime"]).hour == 21), hp[min(9, len(hp)-1)])
        dp = d["properties"]["periods"]
        temps = [p["temperature"] for p in dp[:2]]
        rain_words = ("rain", "shower", "storm", "drizzle")
        issue = next((p["shortForecast"] for p in hp[:12] if any(w in p["shortForecast"].lower() for w in rain_words)), None)

        def mph(s):
            try:
                return int(str(s).split()[0])
            except Exception:
                return 0
        hourly = []
        for p in hp[:16]:
            dpc = (p.get("dewpoint") or {}).get("value")
            hourly.append({"h": datetime.fromisoformat(p["startTime"]).hour, "t": p["temperature"],
                           "dew": round(dpc * 9 / 5 + 32) if dpc is not None else None,
                           "wind": mph(p.get("windSpeed"))})
        return {"now_temp": now["temperature"], "feels": now["temperature"], "cond": now["shortForecast"],
                "high": max(temps), "low": min(temps), "tonight_temp": tonight["temperature"],
                "issue": (f"{issue} later" if issue else "No rain expected"), "ok": True,
                "hourly": hourly}
    except Exception:
        return {"ok": False}


def day_plan(wx: dict) -> str:
    """Turn the hourly curve into one outfit decision for the whole day."""
    if not wx.get("ok") or not wx.get("hourly"):
        return ""
    hrs = wx["hourly"]
    t0 = hrs[0]["t"]
    peak = max(hrs, key=lambda p: p["t"])
    evening = next((p for p in hrs if p["h"] in (21, 22)), hrs[-1])
    def ampm(h):
        return f"{h % 12 or 12}{'pm' if h >= 12 else 'am'}"
    if peak["t"] - t0 >= 8 and peak["t"] - evening["t"] >= 8:
        return (f"It's {t0}° now but {peak['t']}° by {ampm(peak['h'])}, then back to {evening['t']}° tonight. "
                f"Dress for the afternoon and carry a layer for the ride home.")
    if peak["t"] - t0 >= 8:
        return f"It only climbs from here, {peak['t']}° by {ampm(peak['h'])}. Dress for later, not for now."
    if t0 - evening["t"] >= 8:
        return f"This is the warm part of the day. {evening['t']}° by tonight, so whatever you skip now you'll want at 9pm."
    return f"Steady around {t0}° all day. One outfit covers you start to finish."


def advisories(wx: dict) -> list:
    """Audience-native comfort intel from data the cameras can't see."""
    if not wx.get("ok"):
        return []
    out = []
    hrs = wx.get("hourly") or []
    dew = next((p["dew"] for p in hrs if p["dew"] is not None), None)
    wind = max((p["wind"] for p in hrs[:8]), default=0)
    t = wx["now_temp"]
    if dew is not None and dew >= 70:
        out.append(f"Dewpoint {dew}. Full frizz conditions, hair-tie weather.")
    elif dew is not None and dew >= 65:
        out.append(f"A little humid today ({dew} dewpoint). Hair will notice.")
    if wind >= 18:
        out.append(f"Windy on the avenues, around {wind} mph. Maybe not the flowy skirt day.")
    if t >= 85:
        out.append("Subway platforms run about 15 degrees hotter than the street. Dress for the platform.")
    if t >= 88:
        out.append("Every office and store will overcorrect the AC. Keep a desk layer handy.")
    if t <= 25:
        out.append("Overheated stores and subway cars ahead. Layers you can peel beat one big coat.")
    return out[:3]


def yesterday_reference() -> Optional[dict]:
    """Yesterday's sweep nearest to this hour (n>=25), for delta headlines."""
    try:
        y = (now_et() - timedelta(days=1)).strftime("%Y%m%d")
        best = None
        hdir = os.path.join(DATA, "history")
        for f in os.listdir(hdir):
            if f.startswith(y):
                h = json.load(open(os.path.join(hdir, f)))
                if h.get("sampled", 0) >= 25:
                    gap = abs(datetime.fromisoformat(h["generated_at"]).hour - now_et().hour)
                    if best is None or gap < best[0]:
                        best = (gap, h)
        return best[1] if best and best[0] <= 2 else None
    except Exception:
        return None


def overdress_advisory() -> Optional[str]:
    """Did yesterday's morning jackets get abandoned by noon? Then say so this morning."""
    if now_et().hour >= 11:
        return None
    try:
        y = (now_et() - timedelta(days=1)).strftime("%Y%m%d")
        hdir = os.path.join(DATA, "history")
        am, noon = [], []
        for f in os.listdir(hdir):
            if f.startswith(y):
                h = json.load(open(os.path.join(hdir, f)))
                agg = h.get("agg")
                if not agg or h.get("sampled", 0) < 25:
                    continue
                hr = datetime.fromisoformat(h["generated_at"]).hour
                if 7 <= hr <= 10:
                    am.append(agg["jacket_rate"] + agg["hoodie_rate"])
                elif 12 <= hr <= 15:
                    noon.append(agg["jacket_rate"] + agg["hoodie_rate"])
        if am and noon and max(am) >= 0.3 and (max(am) - min(noon)) >= 0.15:
            return "Yesterday's morning layers got carried home by noon. You can probably skip yours."
    except Exception:
        pass
    return None


# ---------- aggregate + spectrum + copy ----------

BANDS = [  # (min_implied_F, name, verdict_noun)
    (95, "scorcher", "as-little-as-possible"), (85, "hot", "tank top"), (75, "warm", "shorts"),
    (65, "pleasant", "t-shirt"), (55, "layered", "light layer"), (45, "brisk", "hoodie"),
    (32, "cold", "real coat"), (-99, "deep winter", "heaviest coat"),
]


def implied_band(agg: dict) -> tuple:
    """Map clothing rates to a band + implied temp. Crude v1; self-calibrates later vs NWS log."""
    n = agg["n"]
    if n == 0:
        return ("unknown", 70)
    shorts = agg["shorts_rate"] + agg["skirt_rate"]
    jackets = agg["jacket_rate"] + agg["hoodie_rate"]
    bare = agg["short_sleeve_rate"] + agg["sleeveless_rate"]
    if agg["sleeveless_rate"] > 0.25 and shorts > 0.6:
        t = 88
    elif shorts >= 0.45 and jackets < 0.15:
        t = 79
    elif bare >= 0.55 and jackets < 0.25:
        t = 72
    elif jackets < 0.5:
        t = 62
    elif jackets < 0.8:
        t = 50
    else:
        t = 38
    band = next(b for b in BANDS if t >= b[0])
    return (band[1], t)


def rate_phrase(k: int, n: int) -> str:
    if n == 0:
        return "no one out yet"
    f = k / n
    if k == 0:
        return "not a single one" if n >= 60 else "none we could spot"
    if k == 1:
        return "just one brave soul"
    if f >= 0.85:
        return "almost everyone"
    if f >= 0.6:
        return "most people"
    if f >= 0.4:
        return "about half"
    if f >= 0.2:
        return "1 in 4 ♥" if 0.2 <= f < 0.3 else "a good third"
    if f >= 0.07:
        return "a few"
    return "a handful"


def aggregate(cam_results: list) -> dict:
    people = [p for r in cam_results for p in r["people"]]
    n = len(people)
    # rates use only confidently-classified people as denominator: "unclear" must not
    # masquerade as long pants / covered arms (systematic-bias fix)
    outer_known = [p for p in people if p["outer"] != "unclear"]
    bott_known = [p for p in people if p["bottoms"] != "unclear"]
    def rate(pool, pred):
        return sum(1 for p in pool if pred(p)) / len(pool) if pool else 0
    def count(pred):
        return sum(1 for p in people if pred(p))
    return {
        "n": n, "cams": len(cam_results),
        "outer_known": len(outer_known), "bottoms_known": len(bott_known),
        "short_sleeve_rate": rate(outer_known, lambda p: p["outer"] == "short_sleeve"),
        "sleeveless_rate": rate(outer_known, lambda p: p["outer"] == "sleeveless"),
        "jacket_rate": rate(outer_known, lambda p: p["outer"] == "jacket_or_coat"),
        "hoodie_rate": rate(outer_known, lambda p: p["outer"] == "hoodie"),
        "shorts_rate": rate(bott_known, lambda p: p["bottoms"] == "shorts"),
        "skirt_rate": rate(bott_known, lambda p: p["bottoms"] == "skirt_or_dress"),
        "pants_rate": rate(bott_known, lambda p: p["bottoms"] == "long_pants"),
        "umbrella_count": count(lambda p: p["umbrella"]),
        "jacket_count": count(lambda p: p["outer"] == "jacket_or_coat"),
        "shorts_count": count(lambda p: p["bottoms"] == "shorts"),
        "skirt_count": count(lambda p: p["bottoms"] == "skirt_or_dress"),
        "tee_count": count(lambda p: p["outer"] in ("short_sleeve", "sleeveless")),
        "colors": dict(__import__("collections").Counter(
            p.get("top_color") for p in people
            if p.get("top_color") and p["top_color"] != "unclear").most_common(6)),
    }


def palette_line(agg: dict) -> Optional[str]:
    colors = agg.get("colors") or {}
    known = sum(colors.values())
    if known < 20:
        return None
    top = sorted(colors.items(), key=lambda kv: -kv[1])
    c1, n1 = top[0]
    if n1 / known >= 0.4:
        return f"mostly {c1} today"
    if len(top) > 1 and (n1 + top[1][1]) / known >= 0.5:
        return f"{c1} & {top[1][0]}, mostly"
    return "a bit of everything"


def build_today(cam_results: list) -> dict:
    agg = aggregate(cam_results)
    wx = nws()
    band, implied = implied_band(agg)
    n = agg["n"]
    hour = now_et().hour
    hello = "Good morning ☀️" if hour < 12 else ("Good afternoon ☀️" if hour < 17 else "Good evening 🌆")

    noun = next((b[2] for b in BANDS if b[1] == band), "t-shirt")

    # verdict priority ladder: thin sample > big delta vs yesterday > band
    yref = yesterday_reference()
    delta = (implied - yref["implied_temp"]) if yref else 0
    if n < 25:
        verdict = "The street's still waking up."
    elif yref and abs(delta) >= 6:
        word = "warmer" if delta > 0 else "cooler"
        verdict = f"Not yesterday: <em>{abs(round(delta))}° {word}</em>."
    else:
        verdict = f"It's a <em>{noun}</em> kind of day."

    sub = ""
    if wx.get("ok") and n >= 25:
        diff = implied - wx["now_temp"]
        if yref and abs(delta) >= 6:
            sub = f"The street already adjusted, <b>{noun}</b> mode out there. Yesterday's outfit won't work today."
        elif diff >= 4:
            sub = f"The forecast says {wx['now_temp']}°, but honestly? Everyone's dressed for <b>warmer</b>. We'd believe them."
        elif diff <= -4:
            sub = f"The forecast says {wx['now_temp']}°, but the street is dressed <b>cozier</b> than that. Worth a listen."
        else:
            sub = f"The street and the forecast agree for once: about <b>{wx['now_temp']}°</b> energy out there."
    elif n < 25:
        sub = "Small crowd so far — here's what the early birds picked."

    rows = [
        {"label": "Tees & short sleeves", "value": rate_phrase(agg["tee_count"], agg["outer_known"])},
        {"label": "Shorts", "value": rate_phrase(agg["shorts_count"], agg["bottoms_known"])},
        {"label": "Skirts & dresses", "value": rate_phrase(agg["skirt_count"], agg["bottoms_known"]), "love": agg["skirt_rate"] >= 0.15},
        {"label": "Umbrellas", "value": rate_phrase(agg["umbrella_count"], n)},
        {"label": "Jackets", "value": rate_phrase(agg["jacket_count"], agg["outer_known"])},
    ]
    pal = palette_line(agg)
    if pal:
        rows.append({"label": "The palette", "value": pal})

    cams_out = []
    for r in cam_results:
        if not r["people"]:
            continue
        cams_out.append({"cam_id": r["cam_id"], "name": pretty_cam(r["cam"]),
                         "people": len(r["people"]),
                         "boxes": [{"box_2d": p["box_2d"], "label": short_label(p)} for p in r["people"]]})
    cams_out.sort(key=lambda c: -c["people"])

    adv = advisories(wx)
    od = overdress_advisory()
    if od:
        adv = [od] + adv
    wx_out = {k: v for k, v in wx.items() if k != "hourly"} if wx.get("ok") else None
    today = {"generated_at": now_et().isoformat(timespec="seconds"),
             "hello": hello, "verdict_html": verdict, "sub_html": sub,
             "band": band, "implied_temp": implied, "sampled": n,
             "weather": wx_out, "plan": day_plan(wx), "advisories": adv[:3],
             "wearing": rows, "cams": cams_out[:6],
             "arc": build_arc(band, noun, wx), "agg": agg}
    return today


def band_noun(temp: int) -> str:
    return next(b[2] for b in BANDS if temp >= b[0])


def build_arc(now_band: str, now_noun: str, wx: dict) -> list:
    """morning = earliest sweep today · now = this sweep · tonight = NWS forecast, not clothes."""
    morning = None
    try:
        stamp_prefix = now_et().strftime("%Y%m%d")
        hist = sorted(f for f in os.listdir(os.path.join(DATA, "history")) if f.startswith(stamp_prefix))
        for f in hist:
            h = json.load(open(os.path.join(DATA, "history", f)))
            if h.get("sampled", 0) >= 25 and datetime.fromisoformat(h["generated_at"]).hour < 11:
                morning = {"t": "THIS MORNING", "v": band_noun(h["implied_temp"]).title(),
                           "d": f"{h['band']} out there"}
                break
    except Exception:
        pass
    if morning is None:
        morning = {"t": "THIS MORNING", "v": "—", "d": "we weren't up yet"}
    now = {"t": "RIGHT NOW", "v": now_noun.title(), "d": f"{now_band} mode", "now": True}
    if wx.get("ok"):
        tn = band_noun(wx["tonight_temp"])
        tonight = {"t": "TONIGHT", "v": tn.title(), "d": f"forecast says {wx['tonight_temp']}°"}
    else:
        tonight = {"t": "TONIGHT", "v": "—", "d": "check back later"}
    return [morning, now, tonight]


def pretty_cam(name: str) -> str:
    return name.replace("@", "&").title().replace("St", "St").replace("Ave", "Ave")


def short_label(p: dict) -> str:
    o = {"jacket_or_coat": "jacket", "hoodie": "hoodie", "long_sleeve": "long sleeve",
         "short_sleeve": "tee", "sleeveless": "tank", "unclear": ""}[p["outer"]]
    b = {"shorts": "shorts", "long_pants": "pants", "skirt_or_dress": "skirt/dress", "unclear": ""}[p["bottoms"]]
    lab = " / ".join(x for x in (o, b) if x)
    return lab + (" / umbrella" if p["umbrella"] else "")


# ---------- state (optional GCS persistence: set GCS_BUCKET) ----------

_gcs = None
def gcs_bucket():
    global _gcs
    name = os.getenv("GCS_BUCKET")
    if not name:
        return None
    if _gcs is None:
        from google.cloud import storage
        _gcs = storage.Client().bucket(name)
    return _gcs


def gcs_put(rel: str):
    try:
        b = gcs_bucket()
        if b:
            b.blob(rel).upload_from_filename(os.path.join(DATA, rel))
    except Exception:
        pass


def gcs_restore():
    """On cold start, pull state back so history/feedback/calibration survive restarts."""
    try:
        b = gcs_bucket()
        if not b:
            return
        for blob in b.list_blobs():
            dest = os.path.join(DATA, blob.name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                blob.download_to_filename(dest)
    except Exception:
        pass


gcs_restore()


def save_today(today: dict):
    with _lock:
        json.dump(today, open(os.path.join(DATA, "today.json"), "w"))
        stamp = now_et().strftime("%Y%m%d-%H%M")
        json.dump(today, open(os.path.join(DATA, "history", f"{stamp}.json"), "w"))
        # daily self-audit log: implied vs official, accumulates calibration data from day one
        if today.get("weather"):
            with open(os.path.join(DATA, "calibration.jsonl"), "a") as f:
                f.write(json.dumps({"ts": today["generated_at"], "implied": today["implied_temp"],
                                    "official": today["weather"]["now_temp"], "n": today["sampled"]}) + "\n")
    for rel in ("today.json", f"history/{stamp}.json", "calibration.jsonl"):
        gcs_put(rel)


def archive_eval_samples(results: list, stamp: str):
    """Archive frame+predictions pairs to the bucket for the accuracy eval pool.
    One busiest cam + one random cam per sweep — a slowly growing, unbiased test set."""
    import random
    import shutil
    with_people = [r for r in results if r["people"]]
    if not with_people:
        return
    picks = {with_people[0]["cam_id"]: with_people[0]}
    for r in sorted(with_people, key=lambda r: -len(r["people"]))[:1] + \
             ([random.choice(with_people)] if len(with_people) > 1 else []):
        picks[r["cam_id"]] = r
    for r in picks.values():
        rel_dir = os.path.join("eval_pool", stamp)
        os.makedirs(os.path.join(DATA, rel_dir), exist_ok=True)
        src = os.path.join(DATA, "frames", f"{r['cam_id']}.jpg")
        if not os.path.exists(src):
            continue
        shutil.copy(src, os.path.join(DATA, rel_dir, f"{r['cam_id']}.jpg"))
        json.dump({"cam": r["cam"], "cam_id": r["cam_id"], "stamp": stamp,
                   "model": os.getenv("DETECT_MODEL", "gemini-flash-lite-latest"),
                   "people": r["people"]},
                  open(os.path.join(DATA, rel_dir, f"{r['cam_id']}.json"), "w"))
        gcs_put(f"{rel_dir}/{r['cam_id']}.jpg")
        gcs_put(f"{rel_dir}/{r['cam_id']}.json")


def run_sweep() -> dict:
    with cf.ThreadPoolExecutor(2) as ex:
        results = [r for r in ex.map(sweep_cam, CAMS) if r]
    today = build_today(results)
    save_today(today)
    try:
        archive_eval_samples(results, now_et().strftime("%Y%m%d-%H%M"))
    except Exception as e:
        print(f"[eval] archive failed: {e}", flush=True)
    return today


# ---------- push notifications ----------

VAPID_SUB = os.getenv("VAPID_SUB", "mailto:destinationaryan@gmail.com")
VAPID_PEM = None
if os.getenv("VAPID_PRIVATE_B64"):
    VAPID_PEM = "/tmp/vapid.pem"
    open(VAPID_PEM, "wb").write(base64.b64decode(os.environ["VAPID_PRIVATE_B64"]))


def _load_subs() -> dict:
    subs = {}
    try:
        for line in open(os.path.join(DATA, "subscriptions.jsonl")):
            s = json.loads(line)
            subs[s["endpoint"]] = s
    except FileNotFoundError:
        pass
    return subs


def _save_subs(subs: dict):
    with open(os.path.join(DATA, "subscriptions.jsonl"), "w") as f:
        for s in subs.values():
            f.write(json.dumps(s) + "\n")
    gcs_put("subscriptions.jsonl")


def _strip(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "")


class SubReq(BaseModel):
    subscription: dict


@app.post("/api/subscribe")
def api_subscribe(req: SubReq):
    sub = req.subscription
    if not str(sub.get("endpoint", "")).startswith("https://"):
        raise HTTPException(400, "bad subscription")
    with _lock:
        subs = _load_subs()
        subs[sub["endpoint"]] = sub
        _save_subs(subs)
    return {"ok": True, "subscribers": len(subs)}


class UnsubReq(BaseModel):
    endpoint: str


@app.post("/api/unsubscribe")
def api_unsubscribe(req: UnsubReq):
    with _lock:
        subs = _load_subs()
        subs.pop(req.endpoint, None)
        _save_subs(subs)
    return {"ok": True}


@app.post("/api/push-morning")
def api_push_morning(request: Request):
    token = os.getenv("SWEEP_TOKEN")
    if token and request.headers.get("x-sweep-token") != token:
        raise HTTPException(403)
    if not VAPID_PEM:
        raise HTTPException(503, "no vapid key configured")
    from pywebpush import webpush, WebPushException
    try:
        today = json.load(open(os.path.join(DATA, "today.json")))
    except Exception:
        raise HTTPException(503, "no sweep yet")
    title = "DailyWear · " + _strip(today.get("verdict_html", "good morning"))
    body = today.get("plan") or _strip(today.get("sub_html", ""))
    if today.get("advisories"):
        body += " · " + today["advisories"][0]
    payload = json.dumps({"title": title, "body": body[:180]})
    sent = pruned = failed = 0
    with _lock:
        subs = _load_subs()
        for ep, sub in list(subs.items()):
            try:
                webpush(subscription_info=sub, data=payload, ttl=3600 * 3,
                        vapid_private_key=VAPID_PEM, vapid_claims={"sub": VAPID_SUB})
                sent += 1
            except WebPushException as e:
                if e.response is not None and e.response.status_code in (404, 410):
                    subs.pop(ep)
                    pruned += 1
                else:
                    failed += 1
            except Exception as e:
                # malformed subscription (bad keys etc.) — drop it, never crash the send run
                print(f"[push] dropping malformed sub: {str(e)[:80]}", flush=True)
                subs.pop(ep)
                pruned += 1
        _save_subs(subs)
    return {"sent": sent, "pruned": pruned, "failed": failed}


# ---------- api ----------

class FeedbackReq(BaseModel):
    text: str


@app.get("/api/today")
def api_today():
    try:
        return json.load(open(os.path.join(DATA, "today.json")))
    except Exception:
        raise HTTPException(503, "no sweep yet")


@app.post("/api/sweep")
def api_sweep(request: Request):
    token = os.getenv("SWEEP_TOKEN")
    if token and request.headers.get("x-sweep-token") != token:
        raise HTTPException(403)
    today = run_sweep()
    return {"sampled": today["sampled"], "cams": len(today["cams"]), "band": today["band"]}


@app.post("/api/feedback")
def api_feedback(req: FeedbackReq):
    text = req.text.strip()[:2000]
    if text:
        with _lock, open(os.path.join(DATA, "feedback.jsonl"), "a") as f:
            f.write(json.dumps({"ts": now_et().isoformat(timespec="seconds"), "text": text}) + "\n")
        gcs_put("feedback.jsonl")
    return {"ok": True}


@app.get("/api/frame/{cam_id}")
def api_frame(cam_id: str):
    path = os.path.join(DATA, "frames", f"{cam_id}.jpg")
    if not os.path.exists(path) or "/" in cam_id or ".." in cam_id:
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/healthz")
def healthz():
    return {"ok": True, "cams": len(CAMS)}


# ---------- self-scheduling loop (for laptop/VM hosting; Cloud Run uses Cloud Scheduler instead) ----------

def _sweep_loop():
    while True:
        now = now_et()
        if 7 <= now.hour <= 20:
            try:
                t = run_sweep()
                print(f"[sweep] {t['generated_at']} n={t['sampled']} band={t['band']}", flush=True)
            except Exception as e:
                print(f"[sweep] failed: {e}", flush=True)
        time.sleep(30 * 60)


if os.getenv("SWEEP_LOOP") == "1":
    threading.Thread(target=_sweep_loop, daemon=True).start()
