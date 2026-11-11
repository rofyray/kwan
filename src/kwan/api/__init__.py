"""REST API (FastAPI). Needs the `api` extra. The application object is `kwan.api.app:app`:

    uvicorn kwan.api.app:app --port 8000

Settings come from the environment: `KWAN_API_KEYS`, `KWAN_STORE`, `KWAN_BUNDLE` (see `kwan.api.runtime`).
"""
