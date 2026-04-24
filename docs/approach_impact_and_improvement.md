# Approach, impact, and improvement (RS2)

## Approach

RS2 is an **eyewear recommendation** stack that:

1. **Face understanding** — Accepts a user selfie (upload or camera), runs **AWS Rekognition** to validate a single face, derives **face geometry**, and classifies a **face shape** (six buckets) so the catalog can be **filtered** to structurally suitable frames.
2. **Rule-based ranking** — Scores remaining candidates with a **weighted blend** of signals: fit vs face shape (tags + prototype embedding), **popularity**, **demographics** (age/gender/lifecycle heuristics), **region** affinity where configured, **color lifecycle** fit, and **last-order** similarity when product ids are provided.
3. **CLIP style channel (optional)** — When the user adds **style text** and/or a **preference / reference image**, the system can score products with **CLIP** against per-product `clip_embedding` and **blend** that preference with the rule-based score (weights configured via environment, not ad hoc tuning in the UI).

The **Streamlit** app is optimized for demos; the **FastAPI** layer supports the same `recommend_from_bytes` core with user context and catalog loading from the database or S3, depending on configuration.

## Current data reality

- **Catalog and embeddings:** Local samples such as `glasses_catalog.json` and placeholder **prototype vectors** are **dummy / illustrative** data. Production would replace these with a real catalog, real popularity and tag fields, and `clip_embedding` (or an equivalent) built from actual product imagery.
- **Purchase and browsing history:** **Historical purchase data is not available** in the current setup. The ranking pipeline *can* use `last_order_product_ids` when supplied (e.g. manually in the UI, via API / DB user context in production), but we are **not** yet feeding a full **past-order timeline** or order analytics warehouse into the model.

Stating this explicitly avoids overstating what the demo can infer from behavior or revenue history today.

## Impact (when the approach is fully supplied with real data)

| Area | Impact |
|------|--------|
| **Fit** | Reduces misfit returns by matching frame geometry and face shape before style preference. |
| **Merchandising** | Popularity, region, and lifecycle signals align recommendations with what sells and for whom. |
| **Style** | CLIP + `clip_embedding` lets “looks like this” and “in this style” work without hand-crafted keyword search over titles. |
| **Operations** | S3-backed catalog and optional API **presigned** images support the same flow in web and app channels. |

## Improvements (roadmap)

1. **Real catalog and signals** — Wire production S3/DB, refresh pipelines for popularity and regional affinities, and maintain `clip_embedding` (or a single embedding model) from **live product photos**.
2. **Order and identity service** — Persist **user → order lines** and expose `last_order_product_ids` (or richer features) from **actual orders**, not manual entry.
3. **Purchase *images* as preference signal** — The same **image encoding path** used for the **style / preference / mood reference** image can be applied to **photos of frames the user already bought** (or wore in a virtual try-on). Concretely: encode one or more **past-purchase product images** with CLIP, **fuse** them (e.g. average or weighted by recency) into a **query vector**, and reuse the existing **cosine vs `clip_embedding`** preference channel. That reuses `encode_user_style` / fusion patterns already in the codebase without inventing a second similarity stack.
4. **Cold start** — When history is sparse, fall back to demographics + region + popularity; as history grows, increase weight on “similar to what you already chose” via CLIP or collaborative signals derived from order ids.

Together, these steps move the system from **demo-grade dummy data** to a **data-backed** recommender where **visual preference** (reference image) and **purchase imagery** (history) are first-class, aligned channels.
