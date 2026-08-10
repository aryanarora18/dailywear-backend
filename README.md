# DailyWear backend

The service behind [dailywear.nyc](https://dailywear.nyc): it reads what New York is actually wearing from the city's public street cameras and turns it into one morning verdict.

## How it works

Every 30 minutes during daylight (Cloud Scheduler), the sweep:

1. Photographs ~30 curated pedestrian-rich NYC DOT cameras (list in `data/cams.json`, ranked by a citywide curation sweep).
2. Sends each frame to Gemini (`gemini-flash-latest`) with a structured schema: per-person bounding box, top layer, bottoms, top color, umbrella, confidence. People below a size floor or confidence floor are dropped. Frames that are too dark to verify are skipped (`scene_ok`).
3. Aggregates across everyone seen. Rates are computed only over confidently classified people, so "unclear" never masquerades as long pants.
4. Fuses the NWS forecast (hourly curve, dewpoint, wind) for the official tile, the day plan sentence, and advisories (frizz, skirt wind, subway platform heat, AC overcorrection).
5. Writes plain language, not percentages. Phrase strength is tied to sample size: a thin sweep says "mostly tees," never "72%". Verdict priority: thin sample, then big change vs yesterday, then band verdict.
6. Serves one JSON at `/api/today` and composes the 7:40am push note (verdict title plus one fused body line).

State (history, calibration log, feedback, push subscriptions, eval pool) lives in a GCS bucket and survives restarts. Every sweep also logs implied temperature vs official temperature to `calibration.jsonl`, so the clothing-to-temperature mapping can be corrected from real data.

## Endpoints

| Endpoint | What |
|---|---|
| `GET /api/today` | The full daily payload: verdict, plan, advisories, wearing rows, arc, camera cards with boxes |
| `POST /api/sweep` | Run a sweep (requires `x-sweep-token` header) |
| `POST /api/feedback` | Free-text user feedback, appended to the bucket |
| `POST /api/subscribe` / `POST /api/unsubscribe` | Web push subscriptions (anonymous endpoint only) |
| `POST /api/push-morning` | Send the morning note to all subscribers (token protected) |
| `GET /api/frame/{cam_id}` | Latest archived frame for a camera |

## Environment

| Var | Purpose |
|---|---|
| `GEMINI_API_KEYS` | Comma-separated Gemini keys (rotates on 429) |
| `DETECT_MODEL` | Vision model, prod uses `gemini-flash-latest` |
| `GCS_BUCKET` | State bucket (`dailywear-state`) |
| `SWEEP_TOKEN` | Protects sweep and push endpoints |
| `VAPID_PRIVATE_B64`, `VAPID_SUB` | Web push signing |
| `SWEEP_LOOP=1` | Optional self-scheduling loop for laptop/VM hosting (Cloud Run uses Scheduler instead) |

## Deploy

```bash
gcloud run deploy dailywear --source . --region us-east1 --allow-unauthenticated \
  --timeout 600 --memory 512Mi \
  --set-env-vars "^|^GEMINI_API_KEYS=...|DETECT_MODEL=gemini-flash-latest|GCS_BUCKET=dailywear-state|SWEEP_TOKEN=...|VAPID_PRIVATE_B64=...|VAPID_SUB=mailto:..."
```

Note the `^|^` delimiter: the env set is replaced wholesale on each deploy and values contain commas and @ signs. Scheduler jobs (`dailywear-sweep` every 30 min 7am to 8pm ET, `dailywear-push` daily 7:40am ET) live in us-east1. All timestamps use `America/New_York` via `now_et()`; Cloud Run's clock is UTC.

## evalkit/ · the accuracy harness

Every sweep archives a couple of frames with predictions attached to `eval_pool/` in the bucket. Then:

```bash
python3 evalkit/pull.py          # sync pool, build manifest
python3 evalkit/label_server.py  # judge at http://localhost:8400/label.html
python3 evalkit/score.py         # confusion matrices, hallucination rate, weak-class flags
python3 evalkit/golden.py        # re-run current prompt/model on labeled frames before shipping changes
python3 evalkit/weekly_report.py # calibration bias by day, sweep health, user feedback
```

Labels accumulate into a permanent golden set. Privacy stance throughout: outfits are counted, faces are never identifiable at camera resolution, no zoomed crops of individuals are published.

## Frontend

Lives at [aryanarora18/dailywear](https://github.com/aryanarora18/dailywear), served by GitHub Pages on dailywear.nyc. It reads `/api/today`, draws detection boxes client side, and registers the push service worker.
