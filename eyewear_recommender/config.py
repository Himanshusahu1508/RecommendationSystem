# Tuning: CLIP + attribute blend (reference project)
WEIGHT_CLIP_SIM = 0.7
WEIGHT_ATTR_MATCH = 0.3

# FAISS pre-retrieve many; app may use a smaller final top_k
FAISS_CANDIDATE_K = 20
FINAL_TOP_K = 4

# CLIP model (ViT-B/32, 512-d; works on CPU / CUDA / MPS)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

COLORS = [
    "rose gold", "gold", "silver", "black", "tortoise", "tortoiseshell", "clear", "navy", "red",
    "blue", "green", "brown", "white", "gray", "grey", "gunmetal", "transparent", "beige", "pink",
    "yellow", "matte", "chrome",
]
STYLES = [
    "minimal", "minimalist", "formal", "casual", "vintage", "bold", "sporty", "classic", "modern",
    "retro", "elegant", "street", "boho", "bohemian", "academic", "executive", "business",
    "sophisticated", "understated", "luxe", "y2k", "punk", "preppy", "kitsch", "festival", "rave",
    "dopamine",
]
SHAPES = [
    "cat eye", "cateye", "geometric", "rectangular", "round", "oval", "square", "wayfarer",
    "aviator", "rimless", "wrap", "wraparound", "oversized", "butterfly", "pilot", "shield", "goggles",
    "d-frame", "browline", "teashade",
]
ALL_VOCAB: list[str] = sorted(
    {*COLORS, *STYLES, *SHAPES},
    key=len,
    reverse=True,
)
