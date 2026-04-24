# RS2 — Eyewear recommendation (face analysis + S3 catalog)

Rule-based face shape, demographics, regional and color-lifecycle signals, plus optional style prompt. Supports a **flat S3 catalog** (`lusmt{productId}_{view}.jpg`) or a JSON/DB catalog for the API.

## Requirements

- **Python 3.12** (see `runtime.txt`)
- **AWS** with access to **Rekognition** (`DetectFaces`) and **S3** (list/get on your catalog bucket)
- macOS / Linux / Windows (use `python` / `py` where the steps say `python3.12`)

## Quick start (new machine)

1. **Clone** this repository and `cd` into the project root.

2. **Create a virtual environment and install dependencies**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install --U pip
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set at least:

   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
   - `S3_CATALOG_BUCKET`, `S3_CATALOG_PREFIX` (e.g. `all_images` if images live under that prefix)

   Never commit `.env` or real keys.

4. **Optional local data** (if you use JSON + API/seed paths, not required for Streamlit S3 mode):

   - `regional_affinity.json` — present in repo as an example
   - `glasses_catalog.json` — for `scripts/seed_from_json.py`

## Run the Streamlit UI (recommended for demos)

From the project root, with `.env` loaded:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open the URL shown (usually http://localhost:8501). Grant **camera** permission if you use the webcam capture.

## Run the FastAPI server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://127.0.0.1:8000/health  
- Recommend: `POST /v1/recommend` (multipart image; see `app/main.py`)

## CLI tools (same venv + `.env`)

```bash
python image_s3_recommend.py /path/to/photo.jpg
python webcam_s3_eyewear.py
```

## Docker (API + Postgres)

```bash
cp .env.example .env   # add AWS + S3 vars; optional passthrough
docker compose up --build
```

Seed the DB when using Postgres (first time):

```bash
docker compose run --rm api python scripts/seed_from_json.py
```

The API container expects `DATABASE_URL` as in `docker-compose.yml`.

## Makefile (macOS / Linux)

```bash
make install   # creates .venv and pip installs
make run-ui    # Streamlit
make run-api   # Uvicorn
```

## Security

- Rotate any keys that were ever committed or shared.
- In production, prefer **IAM roles** for Rekognition/S3 instead of long-lived access keys.

## More detail

See `CONNECTED.txt` for deployment notes, S3 manifest mode, and embedding options.
