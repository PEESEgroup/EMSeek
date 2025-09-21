<p align="center">
  <img src="./samples/Logo.jpg" alt="EMSeek logo" width="420px" />
</p>

<p align="center">
<a href="docs/architecture.md">
<img src="https://img.shields.io/badge/Read-Architecture-1e3a8a?style=for-the-badge" alt="Architecture Doc" />
</a>
<a href="docs/agent-controllers.md">
<img src="https://img.shields.io/badge/Study-Agent%20Manual-2563eb?style=for-the-badge" alt="Agent Manual" />
</a>
<a href="LICENSE">
<img src="https://img.shields.io/badge/License-Apache%202.0-22c55e?style=for-the-badge" alt="License" />
</a>
<a href="https://github.com/PEESE/EMSeek/issues">
<img src="https://img.shields.io/badge/Open-Issues-f97316?style=for-the-badge&logo=github" alt="Open Issues" />
</a>
</p>

# Bridging Electron Microscopy and Materials Analysis with an Autonomous Agentic Platform

## Overview
EMSeek bridges raw electron microscopy (EM) data to actionable materials insights by pairing advanced reasoning with specialised actions. Rather than a monolithic model, EMSeek runs a provenance-tracked multi-agent system where a Maestro planner delegates work to targeted controllers, maintains shared memory, and streams NDJSON progress so every decision is auditable.

### Why EMSeek Matters
- Unifies perception, structure modelling, property inference, and literature reasoning into a reproducible, provenance-tracked workflow for EM.
- Automates complex multi-stage analyses across diverse materials modalities with minimal human intervention.
- Provides audit-ready artefacts, uncertainty calibration, and physical sanity checks to keep researchers in control.
- Scales from exploratory analysis to production by exposing both browser and programmable interfaces, backed by configurable models and data stores.

## Demo
<video src='https://github.com/user-attachments/assets/d13b118a-ea24-41ae-ae28-c93b36374cc3'></video>

## Quick Start

### 1. Clone & Create an Environment
```bash
git clone https://github.com/PEESE/EMSeek.git
cd EMSeek
conda create -n emseek python=3.10 -y
conda activate emseek
# or: python -m venv .venv && source .venv/bin/activate
```

### 2. Install Core Dependencies
Follow PyTorch's [official instructions](https://pytorch.org/get-started/locally/) for your platform (CPU-only is fine). Then install the EMSeek stack:
```bash
pip install flask gunicorn litellm requests numpy pillow opencv-python scikit-image scipy matplotlib tqdm joblib ase pymatgen torchvision
```

**Optional extras**
- Literature Q&A: `pip install paper-qa`
- Segmentation family: `pip install segmentation-models-pytorch`
- Property backends (UMA, MACE, MatterSim): `pip install fairchem mace-torch mattersim`

### 3. Provide API Keys
EMSeek speaks to LLM/MLLM providers via LiteLLM. Export the keys you need (OpenAI-compatible shown below) before launching:
```bash
export OPENAI_API_KEY="your_openai_key"
# Optional providers
export CORE_API_KEY="your_core_key"
export LLM_MODEL="gpt-5-nano"           # overrides cfg.py defaults
export MLLM_MODEL="gpt-4o-mini"         # vision-capable model for captions
```

### 4. Fetch Models & Reference Data
- Place segmentation and property checkpoints under `pretrained/` according to `cfg.py:TASK2MODEL`.
- Add crystal structure libraries to `database/cif_lib/` (CIF files).
- Drop supporting PDFs into `database/papers_lib/` for offline literature review.
- Runtime logs and artefacts land in `history/<user>/<session>/` automatically.

### 5. Launch & Smoke-Test
**Browser UI (development):**
```bash
python app.py
# visit http://localhost:8000
```

**Production-ready gunicorn:**
```bash
gunicorn -w 2 -k gevent -b 0.0.0.0:8000 "app:app"
```

**REST NDJSON request:**
```bash
curl -N -H 'Content-Type: application/json' \
  -d '{"text": "Segment and describe the uploaded EM image.", "files": ["/abs/path/to/image.png"], "model": "general_model"}' \
  http://localhost:8000/api
```

**Python API:**
```python
from emseek.platform import Platform
import cfg

platform = Platform(cfg)
platform.init_agent()

payload = {"text": "Segment EM image", "files": ["samples/oblique_AgBiSb2S6-1cbf0237027e_supercell_24x20x1_dose30000_sampling0.1_iDPC_V3.png"]}
for frame in platform.query_unified(payload):
    print(frame.strip())  # JSON objects per step/final
```

If uploads are rejected with HTTP 413, raise the proxy limit (`client_max_body_size` in Nginx) and optionally export `MAX_UPLOAD_MB` before starting Flask.


## Configuration Surface
All runtime knobs live in `cfg.py`:
- Model identifiers (`LLM_MODEL`, `MLLM_MODEL`, `EMBEDDING_MODEL`).
- Generation limits (`LLM_MAX_TOKENS`, `MLLM_TEMPERATURE`, etc.).
- Data roots (`HISTORY_ROOT`, `CIF_LIB_DIR`, `PDF_FOLDER`).
- Concurrency and streaming heartbeats.

Environment variables override module defaults, so ops teams can configure deployments without editing code.


## Observability & Provenance
- Every agent step logs JSONL traces and writes artefacts under `history/<user>/<session>/` for audit trails.
- Guardian and Scribe attach provenance metadata, curated context, and uncertainty summaries to outputs.
- Session JSON records persist the latest conversations per user, powering history recall in the web UI.


## Contributing
We welcome pull requests for new agents, analysis tools, retrieval pipelines, and documentation. Open an issue to discuss ideas or share reproducible bug reports. Please ensure new features include tests or runnable notebooks when applicable and respect existing logging and provenance patterns.

## License & Citation
This project is released under the [Apache License 2.0](LICENSE).

```bibtex
@misc{chen2025emseek,
  title        = {Bridging Electron Microscopy and Materials Analysis with an Autonomous Agentic Platform},
  author       = {Chen, Guangyao and Yuan, Wenhao and You, Fengqi},
  year         = {2025},
  note         = {CUAISci, Cornell University}
}
```
