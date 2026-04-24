# RS2 — Eyewear recommendation

Personalized eyewear recommendations using a **live camera or upload**, **AWS Rekognition** (face shape, age, gender), and an **S3-backed catalog**.  
Optional **CLIP** style matching merges with the existing **rule-based ranker** via a **hybrid** layer: default **60% preference / 40% face & rules** (adjustable in the UI or API).

---

## How it works

1. **Camera** captures your face → **AWS Rekognition** `DetectFaces` → face shape (rectangle → heart), age range, gender.  
2. **Catalog** is loaded from **S3** (flat `PRODUCTID_VIEWINDEX.jpg` under `Glass/Sunglass/all_images/`, `Glass/Normal_glass/`, etc.) or from a **manifest** when using the API in S3 mode.  
3. **CLIP style** (optional free text + optional **reference / mood image**) — same CLIP model as your `clip_embedding` build; no separate keyword or substring search over the catalog. Install: `pip install -r requirements-optional-clip.txt`, add per-product `clip_embedding`, tune hybrid via `.env` (`HYBRID_W_*`, `CLIP_PREFERENCE_ENABLED`).  
4. **Hybrid score** (when style text and/or a reference image is provided, CLIP is enabled, and products have embeddings):  
   `final = w_preference × clip_preference + w_face × inner_rule_score`  
   where **inner_rule_score** is the full personalized blend **without** the inner `w_prompt` text slot.  
5. **Top N** frames are returned; Streamlit adds **presigned S3 URLs** for product images.

When **no** style input is given, a **single** ranker runs with **no** text/keyword slot (neutral).

---

## Requirements

- **Python 3.12** (see `runtime.txt`) or 3.11+ with a working venv  
- **AWS** with **Rekognition** and **S3** (list/get on the catalog bucket)  
- S3 layout example: `Glass/Normal_glass/`, `Glass/Sunglass/all_images/` (see `.env.example`)

---

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: `AWS_*`, `S3_CATALOG_BUCKET`, and usually **`S3_CATALOG_PREFIX` empty** if `Glass/` is at the bucket root (see comments in `.env.example`).

### Optional CLIP (style embeddings)

```bash
pip install -r requirements-optional-clip.txt
```

Add `clip_embedding` (list of floats) to each product in your catalog (e.g. merge `metadata.json` from `scripts/build_clip_faiss_index.py` into your manifest or rows).

### macOS (OpenMP with PyTorch)

If you see OpenMP / duplicate library errors, set `KMP_DUPLICATE_LIB_OK=TRUE` in `.env`.

---

## Run the app

**Streamlit (recommended for demos):**

```bash
streamlit run streamlit_app.py
```

- Sidebar: **Sunglasses** vs **Normal glasses**, **Top N**, optional quality skip; hybrid / CLIP weights from `.env`.  
- Main: face upload or camera; optional **style description** and/or **inspiration image** for CLIP.  
- **Get recommendations** runs Rekognition + ranking (and CLIP hybrid when configured and style input is present).

**FastAPI:**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`POST /v1/recommend` — multipart `image` (selfie), optional `style_prompt` and `style_reference_image`; hybrid weights and CLIP on/off come from **environment** (`Settings`), not form fields.

---

## Scoring (reference)

| Layer | Role |
|--------|------|
| **Inner rule-based** | Face tag + shape embedding, popularity, demographics, last order, region, color lifecycle; inner **text** slot set to 0 in hybrid so preference is separate. |
| **Preference** | CLIP vs per-product `clip_embedding` (text and/or reference image as query). |
| **Hybrid** | `w_preference` × preference + `w_face` × normalized inner (defaults 0.6 / 0.4 from `Settings` / UI). |

Per-frame **`score_breakdown`** includes a **`hybrid`** object when the hybrid path runs.

---

## Project layout (essentials)

| Path | Role |
|------|------|
| `streamlit_app.py` | UI, hybrid sliders, S3 catalog load |
| `glasses_recommend.py` | `recommend_from_bytes` — Rekognition + ranking + hybrid |
| `ranking_signals.py` | `personalized_rank`, `RankingWeights` |
| `app/preference/` | Hybrid merge + optional `clip_scoring` |
| `eyewear_recommender/` | CLIP (Hugging Face) + FAISS index helpers for offline builds |
| `scripts/build_clip_faiss_index.py` | Build FAISS index + `metadata.json` with `clip_embedding` per item |
| `app/services/s3_flat_catalog.py` | S3 list → product rows |
| `docs/s3_fetch_architecture.md` | S3 fetch diagram |
| `docs/user_journey_and_ranking.md` | User journey + ranking diagram |

---

## Security

- Do not commit `.env`.  
- Prefer **IAM roles** in production.  
- Rotate any keys that were ever shared.

## More

- `docker compose` — API + Postgres (see `docker-compose.yml`).  
- `scripts/seed_from_json.py` / `scripts/seed_from_s3.py` — populate the API catalog DB when not using in-memory S3 flat listing in Streamlit.
