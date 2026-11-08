# YAMNet classifier service

Small FastAPI service that classifies audio with YAMNet and reports how much of it sounds like singing vs. speech.

## Run

```bash
cd server
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```

The model is downloaded from TF Hub/Kaggle on first start (a few seconds, then cached).

## API

`GET /health` -> `{"status": "ok"}` once the model is loaded.

`POST /classify`
- Body: raw little-endian **float32** mono PCM at **16 kHz**, values in [-1, 1]. Content-Type `application/octet-stream`.
- Length: 15,600 samples (~1 s) minimum, 10 s maximum.
- Response:

```json
{
  "singing": 0.02,
  "speech": 0.98,
  "top": [{ "label": "Speech", "score": 0.99 }]
}
```

`singing` / `speech` are sums of the related AudioSet class scores, averaged over the clip.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `YAMNET_MODEL` | `https://tfhub.dev/google/yamnet/1` | TF Hub handle or path to a local SavedModel |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins allowed by CORS |
| `API_KEY` | unset | If set, requests must send a matching `x-api-key` header |

Note: a key placed in browser code is visible to anyone using the app, so treat `API_KEY` as a light guard, not real authentication.
