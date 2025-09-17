import os

# ------------------------ API keys and model choices ------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default LLM (text) and MLLM (vision) models
LLM_MODEL = os.getenv("LLM_MODEL", 'gpt-5-nano')  # e.g. "gpt-5-nano-2025-08-07"
MLLM_MODEL = os.getenv("MLLM_MODEL", 'gpt-5-nano')

# Centralized generation controls for LLM/MLLM calls
# Tip: set env to "none"/"null"/"auto" or leave blank to use LiteLLM defaults.

def _opt_int(env_key: str, default=None):
    v = os.getenv(env_key, None)
    if v is None:
        return default
    s = v.strip().lower()
    if s == "" or s in {"none", "null", "default", "auto"}:
        return None
    try:
        return int(v)
    except Exception:
        return default

def _opt_float(env_key: str, default=None):
    v = os.getenv(env_key, None)
    if v is None:
        return default
    s = v.strip().lower()
    if s == "" or s in {"none", "null", "default", "auto"}:
        return None
    try:
        return float(v)
    except Exception:
        return default

# None here means: do not send this parameter to LiteLLM (use its default)
LLM_MAX_TOKENS = _opt_int("LLM_MAX_TOKENS", default=None)
LLM_TEMPERATURE = _opt_float("LLM_TEMPERATURE", default=None)
MLLM_MAX_TOKENS = _opt_int("MLLM_MAX_TOKENS", default=None)
MLLM_TEMPERATURE = _opt_float("MLLM_TEMPERATURE", default=None)

# Embedding model for vectorization
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# CORE (research API)
CORE_API_KEY = os.getenv("CORE_API_KEY", "")

# ------------------------ Agent/model resources & paths ---------------------
TASK2MODEL = {
    'general_model': 'pretrained/SegMentor/unet_ref.pth',
    'atomic_column_identification': 'pretrained/SegMentor/task1.pth',
    'atom_defects_detection': 'pretrained/SegMentor/task2.pth',
    'single_atom_catalyst_recognition': 'pretrained/SegMentor/task3.pth',
    'nanoparticle_recognition': 'pretrained/SegMentor/task4.pth',
    'irradiated_alloy_defect_analysis': 'pretrained/SegMentor/task5.pth',
    'crystal_forge_seg_model': 'pretrained/CrystalForge/seg.pt',
    'mat_prophet_model': 'pretrained/MatProphet/MOE.pt',
}

# MatProphet ensemble checkpoint directory
MATPROPHET_MOE_CKPT = os.getenv("MATPROPHET_MOE_CKPT", os.path.join('pretrained', 'MatProphet', 'MoE_Ensemble'))

# Optional base model selections for MatProphet sub-predictors
UMA_MODEL_NAME = os.getenv("UMA_MODEL_NAME", "uma-m-1p1")
MACE_MODEL_NAME = os.getenv("MACE_MODEL_NAME", "medium")

# Directory of CIF files for CrystalForge retrieval
CIF_LIB_DIR = os.getenv("CIF_LIB_DIR", "database/cif_lib")
# Default Top-N for CIF retrieval results
CRYSTALFORGE_TOPK = int(os.getenv("CRYSTALFORGE_TOPK", 5))

# Local PDF library folder for ScholarSeeker/PaperQA
PDF_FOLDER = os.getenv("PDF_FOLDER", "database/papers_lib")

# ------------------------ Platform/runtime behavior ------------------------
# Max reasoning steps for orchestration loop
MAX_STEP = int(os.getenv("MAX_STEP", 10))

# Working directories
HISTORY_ROOT = os.getenv("HISTORY_ROOT", "history")

# Final image display: how many images to include in the final answer bubble
# Excludes raw user-uploaded images; prefers images referenced in final text, then recent agent-produced visuals
FINAL_IMAGES_TOPK = int(os.getenv("FINAL_IMAGES_TOPK", 3))

# ------------------------ ScholarSeeker tuning knobs -----------------------
CORE_TOPN = int(os.getenv("CORE_TOPN", 3))
MAX_CORE_ROUNDS = int(os.getenv("MAX_CORE_ROUNDS", 2))
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
LOG_JSON = os.getenv("LOG_JSON", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

# Preload/refresh behavior
PREWARM_PAPERQA = os.getenv("PREWARM_PAPERQA", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
AUTO_REFRESH_ON_QUERY = os.getenv("AUTO_REFRESH_ON_QUERY", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

# Concurrency knobs (0 means auto-tune)
PAPERQA_INIT_CONCURRENCY = int(os.getenv("PAPERQA_INIT_CONCURRENCY", 8))
CORE_SEARCH_CONCURRENCY = int(os.getenv("CORE_SEARCH_CONCURRENCY", 8))
CORE_DL_CONCURRENCY = int(os.getenv("CORE_DL_CONCURRENCY", 8))
IO_WORKERS = int(os.getenv("IO_WORKERS", 8))

# Stream heartbeat interval (seconds) to keep browser/proxy connections alive
# 0 disables periodic heartbeat (only initial heartbeat is sent)
STREAM_HEARTBEAT_INTERVAL_SEC = int(os.getenv("STREAM_HEARTBEAT_INTERVAL_SEC", 10))
