# EMSeek: Autonomous Agents for Electron Microscopy & Materials Analysis  

**EMSeek** is a modular, provenance-tracked multi-agent platform that connects raw electron microscopy (EM) images to materials structures and property insights. It automates end-to-end workflows—segmentation, crystallographic reconstruction, property prediction, literature grounding, and audit reporting—powered by LLM/MLLM orchestration.  

---

## ✨ Key Features  

- **SegMentor** – Reference-guided, all-in-one segmentation  
- **CrystalForge** – Mask-aware EM → CIF reconstruction  
- **MatProphet** – Gated Mixture-of-Experts property predictor with uncertainty calibration  
- **ScholarSeeker** – Literature retrieval with citation anchoring  
- **Guardian + Scribe** – Physical consistency checks and audit-ready reporting  

---


## ⚙️ Installation  

**Requirements:** Python 3.10+, Linux/macOS preferred. GPU optional (PyTorch auto-detects CUDA).  

1. Create environment:  
```bash
conda create -n emseek python=3.10 -y && conda activate emseek
# or
python -m venv .venv && source .venv/bin/activate
```

2. Install PyTorch (CPU/CUDA): [official guide](https://pytorch.org/get-started/locally/)  

3. Install dependencies:  
```bash
pip install flask gunicorn litellm requests numpy pillow opencv-python scikit-image scipy matplotlib tqdm joblib ase pymatgen torchvision
```

**Optional packages:**  
- Literature Q&A: `pip install paper-qa`  
- Segmentation models: `pip install segmentation-models-pytorch`  
- Property backends (UMA/MACE/MatterSim): `pip install fairchem mace-torch mattersim`  

4. Set API keys:  
```bash
export OPENAI_API_KEY=your_key_here
# optional
export CORE_API_KEY=your_key_here
```

---

## 📦 Data & Models  

- Download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1ltlPT8bclLc9QXfSOEWyKm64ZdtSPxQM?usp=sharing) and place them in `pretrained/` (see `cfg.py:TASK2MODEL`).  
- Place `.cif` files in `database/cif_lib/`.  
- Place PDFs in `database/papers_lib/`.  

Artifacts & logs are written to:  
```
history/<user_id>/
 ├── logs/       # JSONL traces
 └── artifacts/  # masks, overlays, CIFs, patches, etc.
```

---

## 🚀 Usage  

### Web UI (dev mode)  
```bash
python app.py
# visit http://localhost:8000
```

### Production (gunicorn)  
```bash
sh run.sh
# binds to 0.0.0.0:8000
```

### Avoiding 413 (Upload Too Large)  
If you see “Upload too large (413)” when sending image-only requests, your reverse proxy is likely rejecting the request body. Increase the proxy limit and (optionally) Flask’s content length:

- Nginx: set a larger limit in http/server/location blocks and reload Nginx:
  - `client_max_body_size 50M;`
  - `proxy_read_timeout 600s;`
  - See `deploy/nginx.conf.example` for a complete example.
- Flask: the app honors `MAX_UPLOAD_MB` (default 100). Example: `export MAX_UPLOAD_MB=200` before starting.

Alternatively, downscale or compress images before uploading to reduce payload size.

### REST API (NDJSON streaming)  
```bash
curl -N -H 'Content-Type: application/json'   -d '{
        "text": "Segment image and explain",
        "files": ["/abs/path/to/em.png"],
        "model": "general_model"
      }'   http://localhost:8000/api
```

### Python API  
```python
from emseek.platform import Platform
import cfg

plat = Platform(cfg)
plat.init_agent()

payload = {"text": "Segment EM image", "files": ["em.png"]}
for line in plat.query_unified(payload):
    print(line.strip())  # JSON objects (step/final)
```

---

## 🔧 LLM/MLLM Generation Settings

Configure token and temperature limits centrally in `cfg.py` (overridable via environment variables):

- LLM_MAX_TOKENS (env: `LLM_MAX_TOKENS`, default: 10240)
- LLM_TEMPERATURE (env: `LLM_TEMPERATURE`, default: 1.0)
- MLLM_MAX_TOKENS (env: `MLLM_MAX_TOKENS`, default: 10240)
- MLLM_TEMPERATURE (env: `MLLM_TEMPERATURE`, default: 1.0)

All internal `llm`/`mllm` calls respect these settings unless explicitly overridden at the call site.

---

## 🧩 Core Agents  

- **MaestroAgent** – Orchestration & planning  
- **SegMentorAgent** – Segmentation + visual LLM explanation  
- **CrystalForgeAgent** – EM→CIF retrieval with similarity scoring  
- **MatProphetAgent** – MoE property prediction with uncertainty signals  
- **ScholarSeekerAgent** – PaperQA with CORE/local fallback  
- **AnalyzerHubAgent** – Unified tool routing (HyperSpy, py4DSTEM, pymatgen, etc.)  
- **GuardianAgent** – Physical consistency & audit checks  
- **ScribeAgent** – Structured final report  

---

## 📜 License  

This project is licensed under the terms in the [LICENSE](./LICENSE) file.  

---

## 📖 Citation  

If EMSeek is useful for your work, please cite:  

```bibtex
@misc{chen2025emseek,
  title        = {Bridging Electron Microscopy and Materials Analysis with an Autonomous Agentic Platform},
  author       = {Chen, Guangyao and Yuan, Wenhao and You, Fengqi},
  year         = {2025},
  note         = {CUAISci, Cornell University}
}
```
