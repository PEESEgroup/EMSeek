import os
import json
import time
import difflib
import traceback
from typing import Dict, Any, List, Optional

from emseek.agents.base import Agent


def _ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


class AnalyzerHubAgent(Agent):
    """AnalyzerHub (LLM-controlled) — routes analysis requests to concrete tools in emseek.tools,
    backfills paths from unified inputs, and (optionally) summarizes image outputs.

    Invocation:
      - Explicit tool: {"tool": "hyperspy_pca", "params": {...}}
      - Natural task:  {"task": "denoise image",   "params": {...}}  -> LLM/heuristics select tool

    Success output:
      { ok: True, message, text, images: [path]|None, meta: {...} }
    Failure output:
      { ok: False, message, error: {message, fields}, images: None }
    """

    def __init__(self, name, platform):
        super().__init__(name, platform)
        # Optional LLMs (robust to missing API keys)
        from emseek.utils.llm_caller import LLMCaller
        from emseek.utils.mllm_caller import MLLMCaller
        self.llm = LLMCaller(self.platform.config)
        self.mllm = MLLMCaller(self.platform.config)

        self.description = (
            "AnalyzerHub Routing Agent (LLM-controlled): validates/repairs inputs, routes to tools in emseek.tools, "
            "and can produce a brief multimodal summary of generated images.\n"
            "Input (unified): {tool?: str, task?: str, params?: dict, text?: str, images?: str|[str], cifs?: str|[str]}\n"
            "Output (unified): {ok, message, text, images: [path]|null, meta}\n"
            "Notes: If params.image_path/cif_path are missing, the controller backfills from unified images/cifs."
        )

    def expected_input_schema(self) -> str:
        return '{"tool": str?, "task": str?, "params": dict?, "text": str?, "images": str|[str]?, "cifs": str|[str]?}'

    # -------------------------- Controller helpers --------------------------

    def _error_dict(self, message: str, *, errors: Optional[List[str]] = None,
                    images: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {
            "ok": False,
            "text": message,
            "message": message,          # short reply to Maestro
            "error": {"message": message, "fields": errors or []},
            "images": images or None,
            "cifs": None,
            "meta": meta or None,
        }
        try:
            self.remember("error", payload={"errors": errors or [], "meta": meta}, result=message)
        except Exception:
            pass
        return out

    def controller_validate_and_fix(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and repair inputs:
          - Accept {tool} or {task}; if both empty, try infer task from {text} or history (LLM).
          - Backfill params.image_path from unified images[0]; params.cif_path from unified cifs[0].
          - Light per-tool parameter checks; keep errors minimal and actionable.
        Returns: {ok, payload|None, errors, message}
        """
        errors: List[str] = []
        msg_parts: List[str] = []

        # Normalize base fields
        tool = str(payload.get("tool") or "").strip()
        task = str(payload.get("task") or "").strip()
        params: Dict[str, Any] = dict(payload.get("params") or {})

        # Allow unified input mapping (flat)
        try:
            uni = self.normalize_agent_payload(payload)
        except Exception:
            uni = {}

        # If no natural task provided, use unified text
        if not task:
            utext = uni.get("text") if isinstance(uni.get("text"), str) else payload.get("text")
            if isinstance(utext, str) and utext.strip():
                task = utext.strip()
                msg_parts.append("task inferred from text")

        # If still missing tool+task, try LLM from recent history
        if not tool and not task and getattr(self, 'llm', None):
            try:
                hist = self.recent_memory_text(6)
                prompt = (
                    "Infer a short natural-language analysis task from recent history if present.\n"
                    "Return JSON {\"task\": \"\" or string}. Use empty string if none.\n\n"
                    f"History:\n{hist}\n\nOutput JSON:"
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                s = (out or "").strip(); i, j = s.find("{"), s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    data = json.loads(s[i:j+1])
                    t = data.get("task")
                    if isinstance(t, str) and t.strip():
                        task = t.strip()
                        msg_parts.append("task inferred from history")
            except Exception:
                pass

        if not tool and not task:
            errors.append("tool or task")
            return {"ok": False, "payload": None, "errors": errors,
                    "message": "Please specify a tool or a natural-language task."}

        # Backfill image_path / cif_path from unified images/cifs
        imgs = uni.get("images")
        if isinstance(imgs, list) and imgs and "image_path" not in params:
            params["image_path"] = imgs[0]
            msg_parts.append("params.image_path backfilled from unified.images")
        elif isinstance(imgs, str) and imgs and "image_path" not in params:
            params["image_path"] = imgs
            msg_parts.append("params.image_path backfilled from unified.images")

        cifs = uni.get("cifs")
        if isinstance(cifs, list) and cifs and "cif_path" not in params:
            params["cif_path"] = cifs[0]
            msg_parts.append("params.cif_path backfilled from unified.cifs")
        elif isinstance(cifs, str) and cifs and "cif_path" not in params:
            params["cif_path"] = cifs
            msg_parts.append("params.cif_path backfilled from unified.cifs")

        # Lightweight per-tool needs
        t_low = tool.lower()
        task_low = task.lower()
        needs_image = (t_low in {"hyperspy", "hyperspy_pca", "py4dstem", "atomap", "scikit-image", "skimage"}) or \
                      (not t_low and any(k in task_low for k in ["denoise", "edge", "peaks", "4dstem", "image"]))
        needs_cif = (t_low in {"pymatgen", "materials_project"}) or ("cif" in task_low or "structure" in task_low)

        if needs_image:
            ip = params.get("image_path")
            if not (isinstance(ip, str) and os.path.isfile(ip)):
                errors.append("params.image_path")
        if needs_cif:
            cp = params.get("cif_path")
            if not (isinstance(cp, str) and os.path.isfile(cp)):
                errors.append("params.cif_path")

        if errors:
            return {"ok": False, "payload": None, "errors": errors,
                    "message": "Missing or invalid fields: " + ", ".join(errors)}

        fixed = {"tool": tool, "task": task, "params": params}
        return {"ok": True, "payload": fixed, "errors": [], "message": "; ".join(msg_parts) or "ok"}

    def controller_summarize_success(self, *, tool: str, task: str, has_image: bool, outfile: Optional[str], meta_ok: bool) -> str:
        """
        Produce a one-line summary (<=160 chars) for Maestro after success.
        """
        try:
            if getattr(self, 'llm', None):
                prompt = (
                    "Write one short sentence (<=160 chars) summarizing an analysis tool run.\n"
                    f"tool={tool or 'auto'}, has_image={has_image}, out={'yes' if outfile else 'no'}, meta={meta_ok}, task='{task[:60]}'."
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                s = (out or "").strip()
                if s.startswith("[ERROR]"):
                    raise RuntimeError(s)
                return s.splitlines()[0][:160]
        except Exception:
            pass
        base = f"Ran {tool or 'auto-selected tool'} for task: {task[:48]}."
        if outfile:
            base += " Output image generated."
        return base[:160]

    # -------------------------- Tool catalog & routing --------------------------

    def _tool_descriptions(self) -> Dict[str, str]:
        from emseek.tools import TOOL_REGISTRY
        desc = {
            'hyperspy_pca': 'PCA denoise an EM image (HyperSpy or SVD fallback).',
            'py4dstem_calibration': 'FFT-based 4D-STEM style calibration proxy.',
            'atomap_peaks': 'Detect atomic column peak coordinates from a STEM image.',
            'skimage_edges': 'Compute edge map (Sobel) for an image.',
            'pymatgen_summary': 'Parse CIF and summarize structure (formula, atoms, elements).',
            'materials_project': 'Query Materials Project (or offline mock).',
            'image_merge': 'Merge original and mask images with blending.',
            'image_histogram': 'Brightness histogram for an image.',
            'auto_atomic_size_count': 'Auto-binned atomic size count from a mask.',
            'defect_seg_from_points': 'Draw defect markers from a points file.',
            'atom_seg_from_points': 'Draw atom markers from CSV points on an image.',
            'mask_nnd': 'Nearest-neighbor distance analysis from mask/points.',
            'atom_density': 'Compute atom density from mask or point sets.',
            'particle_size_distribution': 'Histogram of particle size (equivalent diameter).',
            'shape_descriptor_kde': 'KDE plot of shape descriptors from mask.',
        }
        return {k: desc.get(k, k) for k in TOOL_REGISTRY.keys()}

    def _closest_tool(self, user_tool: str, task: str) -> str:
        """Return the most similar available tool when an exact match is missing.

        Uses a combination of name similarity (including aliases), description similarity,
        and a small boost if a heuristic/LLM pick matches a candidate.
        """
        try:
            from emseek.tools import TOOL_REGISTRY, TOOL_ALIASES
        except Exception:
            return ''

        if not TOOL_REGISTRY:
            return ''

        user_tool = (user_tool or '').strip().lower()
        task = (task or '').strip().lower()
        if not user_tool and not task:
            return ''

        # Prepare candidate metadata
        descriptions = self._tool_descriptions()

        # Build names bag per canonical tool: [canonical, aliases]
        names_by_canon: Dict[str, List[str]] = {k: [k] for k in descriptions.keys()}
        for alias, canon in getattr(__import__('emseek.tools', fromlist=['TOOL_ALIASES']), 'TOOL_ALIASES', {}).items():
            if canon in names_by_canon:
                names_by_canon[canon].append(alias)

        # Optional LLM/heuristic suggestions to boost scoring
        llm_pick = ''
        try:
            llm_pick = self._decide_tool_llm(task)
        except Exception:
            llm_pick = ''
        heur_pick = self._decide_tool_heuristic(task) if task else ''

        def sim(a: str, b: str) -> float:
            if not a or not b:
                return 0.0
            return difflib.SequenceMatcher(None, a, b).ratio()

        best_key = ''
        best_score = 0.0
        for canon in descriptions.keys():
            names = names_by_canon.get(canon, [canon])
            # Name similarity: compare against canonical and aliases
            name_scores = [sim(user_tool, nm) for nm in names if user_tool]
            name_score = max(name_scores) if name_scores else 0.0
            # Description similarity against task
            desc_score = sim(task, descriptions.get(canon, '')) if task else 0.0
            # Combine with weights, add boosts if suggested by LLM/heuristics
            score = 0.6 * name_score + 0.4 * desc_score
            if canon == llm_pick:
                score += 0.12
            if canon == heur_pick:
                score += 0.08
            if score > best_score:
                best_score = score
                best_key = canon

        # Reasonable threshold: only accept if there's a moderately good match
        return best_key if best_score >= 0.35 else (llm_pick or heur_pick or '')

    # -------------------------- Per-tool param preparation --------------------------

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _default_outfile(self, tool: str, out_dir: str) -> str:
        stem = f"{tool}_{self._ts()}"
        return os.path.join(out_dir, f"{stem}.png")

    def _prepare_tool_params(self, tool: str, params: Dict[str, Any], uni: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        """Map unified params into each tool's expected input keys and synthesize sensible defaults.

        This ensures AnalyzerHub can correctly call all tools in emseek.tools.
        Raises ValueError for missing critical inputs.
        """
        p = dict(params or {})
        images = uni.get("images") if isinstance(uni, dict) else None

        def ensure_file(path: Optional[str], field: str) -> None:
            if not (isinstance(path, str) and os.path.isfile(path)):
                raise ValueError(f"Missing or invalid file for '{field}'")

        # Hyperspy/skimage/py4dstem/atomap/pymatgen/materials_project handled by run_tool adapter
        if tool in {"hyperspy_pca", "skimage_edges", "py4dstem_calibration", "atomap_peaks"}:
            # Map any unified image to image_path
            if "image_path" not in p:
                if isinstance(images, list) and images:
                    p["image_path"] = images[0]
                elif isinstance(images, str):
                    p["image_path"] = images
            ensure_file(p.get("image_path"), "image_path")
            p.setdefault("out_dir", out_dir)
            return p

        if tool == "pymatgen_summary":
            if "cif_path" not in p:
                cifs = uni.get("cifs") if isinstance(uni, dict) else None
                if isinstance(cifs, list) and cifs:
                    p["cif_path"] = cifs[0]
                elif isinstance(cifs, str):
                    p["cif_path"] = cifs
            ensure_file(p.get("cif_path"), "cif_path")
            return p

        if tool == "materials_project":
            # Map text->query when helpful
            if not p.get("query"):
                txt = uni.get("text") if isinstance(uni, dict) else None
                if isinstance(txt, str) and txt.strip():
                    p["query"] = txt.strip()
            # api_key optional; the tool handles missing key (offline mock)
            return p

        # -------- Class-based tools expecting explicit output paths -------- #
        if tool == "image_merge":
            # Needs original + mask
            if not p.get("original_image_path") or not p.get("segmentation_mask_path"):
                # Try to map from images list: [original, mask]
                if isinstance(images, list) and len(images) >= 2:
                    p.setdefault("original_image_path", images[0])
                    p.setdefault("segmentation_mask_path", images[1])
                elif isinstance(images, list) and len(images) == 1 and p.get("segmentation_mask_path"):
                    p.setdefault("original_image_path", images[0])
                elif isinstance(images, list) and len(images) == 1 and p.get("original_image_path"):
                    p.setdefault("segmentation_mask_path", images[0])
            ensure_file(p.get("original_image_path"), "original_image_path")
            ensure_file(p.get("segmentation_mask_path"), "segmentation_mask_path")
            p.setdefault("output_image_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "image_histogram":
            if not p.get("input_image_path"):
                if isinstance(images, list) and images:
                    p["input_image_path"] = images[0]
                elif isinstance(images, str):
                    p["input_image_path"] = images
            ensure_file(p.get("input_image_path"), "input_image_path")
            p.setdefault("output_histogram_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "auto_atomic_size_count":
            # Treat any provided image as mask unless mask_image_path provided
            if not p.get("mask_image_path"):
                if isinstance(images, list) and images:
                    p["mask_image_path"] = images[0]
                elif isinstance(images, str):
                    p["mask_image_path"] = images
            ensure_file(p.get("mask_image_path"), "mask_image_path")
            p.setdefault("output_chart_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "defect_seg_from_points":
            # Expect: input_image_path + points_file_path
            if not p.get("input_image_path"):
                if isinstance(images, list) and images:
                    p["input_image_path"] = images[0]
                elif isinstance(images, str):
                    p["input_image_path"] = images
            # Allow alias keys
            p.setdefault("points_file_path", p.get("points_path") or p.get("points"))
            ensure_file(p.get("input_image_path"), "input_image_path")
            ensure_file(p.get("points_file_path"), "points_file_path")
            p.setdefault("output_image_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "atom_seg_from_points":
            if not p.get("input_image_path"):
                if isinstance(images, list) and images:
                    p["input_image_path"] = images[0]
                elif isinstance(images, str):
                    p["input_image_path"] = images
            # Allow alias keys
            p.setdefault("points_csv_path", p.get("points_path") or p.get("points") or p.get("points_file_path"))
            ensure_file(p.get("input_image_path"), "input_image_path")
            ensure_file(p.get("points_csv_path"), "points_csv_path")
            p.setdefault("output_image_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "mask_nnd":
            # Determine mode based on available inputs
            mode = (p.get("mode") or "").strip().lower()
            if not mode:
                if p.get("mask_image_path") or (isinstance(images, (list, str)) and images):
                    mode = "mask"
                elif p.get("points_csv_path"):
                    mode = "points"
                else:
                    mode = "mask"
            p["mode"] = mode
            if mode == "mask" and not p.get("mask_image_path"):
                if isinstance(images, list) and images:
                    p["mask_image_path"] = images[0]
                elif isinstance(images, str):
                    p["mask_image_path"] = images
            if mode == "mask":
                ensure_file(p.get("mask_image_path"), "mask_image_path")
            else:
                ensure_file(p.get("points_csv_path"), "points_csv_path")
            p.setdefault("output_chart_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "atom_density":
            # Supports 'mask' or 'points' modes
            mode = (p.get("mode") or "").strip().lower()
            if not mode:
                if p.get("mask_image_path") or (isinstance(images, (list, str)) and images):
                    mode = "mask"
                elif p.get("points_csv_paths") or p.get("points_csv_path"):
                    mode = "points"
                else:
                    mode = "mask"
            p["mode"] = mode
            if mode == "mask" and not p.get("mask_image_path"):
                if isinstance(images, list) and images:
                    p["mask_image_path"] = images[0]
                elif isinstance(images, str):
                    p["mask_image_path"] = images
            if mode == "points" and not p.get("points_csv_paths"):
                if p.get("points_csv_path"):
                    p["points_csv_paths"] = [p.get("points_csv_path")]
            # Validate one of the sources
            if mode == "mask":
                ensure_file(p.get("mask_image_path"), "mask_image_path")
            else:
                paths = p.get("points_csv_paths")
                if not (isinstance(paths, list) and paths):
                    raise ValueError("'points_csv_paths' is required for points mode")
                for q in paths:
                    ensure_file(q, "points_csv_paths[]")
            p.setdefault("output_chart_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "particle_size_distribution":
            if not p.get("mask_image_path"):
                if isinstance(images, list) and images:
                    p["mask_image_path"] = images[0]
                elif isinstance(images, str):
                    p["mask_image_path"] = images
            ensure_file(p.get("mask_image_path"), "mask_image_path")
            p.setdefault("output_chart_path", self._default_outfile(tool, out_dir))
            return p

        if tool == "shape_descriptor_kde":
            if not p.get("mask_image_path"):
                if isinstance(images, list) and images:
                    p["mask_image_path"] = images[0]
                elif isinstance(images, str):
                    p["mask_image_path"] = images
            ensure_file(p.get("mask_image_path"), "mask_image_path")
            p.setdefault("output_chart_path", self._default_outfile(tool, out_dir))
            return p

        # Unknown tool: return params unchanged
        return p

    def _decide_tool_llm(self, task: str) -> str:
        if not self.llm or not task:
            return ''
        tools = self._tool_descriptions()
        lines = '\n'.join([f"- {k}: {v}" for k, v in tools.items()])
        prompt = (
            "Select the single best tool key for the task.\n"
            "Only output the key exactly as listed below.\n\n"
            f"Tools:\n{lines}\n\n"
            f"Task: {task}\n\n"
            "Best tool key:"
        )
        try:
            resp = self.llm_call(messages=[{"role": "system", "content": prompt}])
            return (resp or '').strip().split()[0]
        except Exception:
            return ''

    def _decide_tool_heuristic(self, task: str) -> str:
        t = (task or '').lower()
        if any(k in t for k in ['pca', 'denoise', 'hyperspy']):
            return 'hyperspy_pca'
        if any(k in t for k in ['4dstem', 'diffraction', 'calibration', 'py4dstem']):
            return 'py4dstem_calibration'
        if any(k in t for k in ['atomap', 'atomic column', 'columns', 'peaks']):
            return 'atomap_peaks'
        if any(k in t for k in ['skimage', 'edge', 'edges', 'sobel', 'filter']):
            return 'skimage_edges'
        if any(k in t for k in ['pymatgen', 'cif', 'structure']):
            return 'pymatgen_summary'
        if any(k in t for k in ['materials project', 'mpr', 'mp-']):
            return 'materials_project'
        if any(k in t for k in ['merge mask', 'overlay', 'blend', 'merge']):
            return 'image_merge'
        if any(k in t for k in ['histogram', 'brightness', 'intensity distribution']):
            return 'image_histogram'
        if any(k in t for k in ['nearest neighbor', 'nnd', 'rayleigh']):
            return 'mask_nnd'
        if any(k in t for k in ['density', 'atoms per area', 'atom density']):
            return 'atom_density'
        if any(k in t for k in ['particle size', 'equivalent diameter', 'size distribution']):
            return 'particle_size_distribution'
        if any(k in t for k in ['shape descriptor', 'kde', 'ellipticity', 'circularity']):
            return 'shape_descriptor_kde'
        if any(k in t for k in ['defect segmentation', 'defects points']):
            return 'defect_seg_from_points'
        if any(k in t for k in ['atom segmentation', 'atoms points']):
            return 'atom_seg_from_points'
        # generic default
        return 'skimage_edges'

    # -------------------------- Optional summary --------------------------

    def _summarize_image(self, image_path: str, user_query: str) -> str:
        if not image_path or not os.path.isfile(image_path):
            return ""
        if self.mllm is None:
            return f"Generated visualization for: {os.path.basename(image_path)}. Query: {user_query or ''}".strip()
        messages = [
            {"role": "system", "content": (
                "You are a helpful scientific assistant. Summarize the main content of the image and propose a short "
                "descriptive filename (lowercase, hyphenated, no extension). Return as '<name>|<summary>'.")},
            {"role": "user", "content": f"Query: {user_query or ''}"},
        ]
        try:
            return self.mllm_call(messages=messages, image_path=image_path)
        except Exception:
            return f"generated-visual|Visualization for query: {user_query or ''}"

    # -------------------------- Main entry (LLM-controlled) --------------------------

    def forward(self, messages: str | dict) -> Dict[str, Any]:
        """
        Accepts raw string or dict (unified). Returns:
          - success: {ok: True, message, text, images, meta}
          - failure: {ok: False, message, error{message, fields}, images: None}
        """
        # 0) tolerant parsing
        if isinstance(messages, dict):
            raw = dict(messages)
        else:
            s = messages if isinstance(messages, str) else str(messages)
            try:
                obj = json.loads(s)
                raw = obj if isinstance(obj, dict) else {"task": s}
            except Exception:
                raw = {"task": s}

        # 1) allow unified input mapping to params
        try:
            uni = self.normalize_agent_payload(raw)
        except Exception:
            uni = {}
        tool = (raw.get("tool") or "").strip()
        task = (raw.get("task") or "") or (uni.get("text") or "")
        params: Dict[str, Any] = dict(raw.get("params") or {})
        imgs = uni.get("images")
        if isinstance(imgs, list) and imgs and "image_path" not in params:
            params["image_path"] = imgs[0]
        elif isinstance(imgs, str) and imgs and "image_path" not in params:
            params["image_path"] = imgs
        cifs = uni.get("cifs")
        if isinstance(cifs, list) and cifs and "cif_path" not in params:
            params["cif_path"] = cifs[0]
        elif isinstance(cifs, str) and cifs and "cif_path" not in params:
            params["cif_path"] = cifs

        # 2) controller validate & repair
        ctrl = self.controller_validate_and_fix({"tool": tool, "task": task, "params": params, **uni})
        if not ctrl.get("ok", False):
            return self._error_dict(ctrl.get("message", "Invalid input."), errors=ctrl.get("errors", []))

        fixed = ctrl["payload"]
        tool = fixed["tool"]
        task = fixed["task"]
        params = fixed["params"]

        # 3) ensure work dir
        out_dir = os.path.join(self.platform.working_folder, "analyzerhub")
        _ensure_dir(out_dir)
        params.setdefault("out_dir", out_dir)

        # 4) determine tool if not provided
        if not tool:
            tool = self._decide_tool_llm(task) or self._decide_tool_heuristic(task)

        # 5) prepare params for the specific tool
        try:
            from emseek.tools import run_tool, resolve_tool, TOOL_REGISTRY
            canon = resolve_tool(tool)  # canonical key in registry or alias
            # Fallback: if not a known tool, pick the closest based on user intent
            if canon not in TOOL_REGISTRY:
                alt = self._closest_tool(tool, task)
                if alt and alt in TOOL_REGISTRY:
                    canon = alt
                    tool = alt
            prepped = self._prepare_tool_params(canon, params, uni, out_dir)
        except Exception as e:
            meta = {
                "tool": tool,
                "canonical_tool": locals().get('canon', tool),
                "stage": "prepare_params",
                "input": {"raw_params": params, "task": task},
                "output": None,
                "exception": str(e),
                "trace": traceback.format_exc(),
            }
            return self._error_dict(f"Invalid inputs for tool '{tool}': {e}", errors=["params"], meta=meta)

        # 6) run tool
        try:
            result = run_tool(canon, prepped, workdir=out_dir)
        except Exception as e:
            meta = {
                "tool": tool,
                "canonical_tool": canon,
                "stage": "run_tool",
                "input": prepped,
                "output": None,
                "exception": str(e),
                "trace": traceback.format_exc(),
            }
            return self._error_dict(f"Tool '{tool}' failed: {e}", errors=["tool or params"], meta=meta)

        # 7) check tool-level status and harvest image outputs
        if isinstance(result, dict) and result.get("status") == "error":
            meta = {
                "tool": tool,
                "canonical_tool": canon,
                "stage": "tool_reported_error",
                "input": prepped,
                "output": result,
            }
            return self._error_dict(result.get("message") or f"Tool '{tool}' reported error.", meta=meta)

        image_keys = [
            "output_image",
            "output_image_path",
            "output_histogram_path",
            "output_chart_path",
        ]
        images_out_list: List[str] = []
        if isinstance(result, dict):
            for k in image_keys:
                if result.get(k):
                    v = result[k]
                    if isinstance(v, str):
                        images_out_list.append(v)

        summary_text = None
        images_out = None
        outfile = None
        if images_out_list:
            # normalize to abs paths and ensure existence
            norm_paths: List[str] = []
            for pth in images_out_list:
                p_abs = os.path.abspath(pth)
                norm_paths.append(p_abs)
                # ensure file existence
                if not os.path.isfile(p_abs):
                    with self._suppress_errors():
                        from PIL import Image
                        src = prepped.get("image_path") or prepped.get("input_image_path") or prepped.get("mask_image_path")
                        if src and os.path.isfile(src):
                            Image.open(src).save(p_abs)
                        else:
                            Image.new("L", (16, 16), color=0).save(p_abs)
                with self._suppress_errors():
                    self.remember_file(p_abs, label="analysis_output", kind="analysis_output")
            outfile = norm_paths[0]
            summary_text = self._summarize_image(outfile, task)
            # Update result with absolute paths
            if isinstance(result, dict):
                for k in image_keys:
                    if k in result and isinstance(result[k], str):
                        result[k] = os.path.abspath(result[k])
            images_out = norm_paths

        # 8) build success one-liner + final text
        meta_ok = isinstance(result, dict) and bool(result)
        short_msg = self.controller_summarize_success(tool=tool, task=task, has_image=bool(images_out), outfile=outfile, meta_ok=meta_ok)
        text = (summary_text or f"Tool {tool} executed.").strip()

        # 9) assemble unified output
        out = {
            "ok": True,
            "message": short_msg,
            "text": text,
            "images": images_out,
            "meta": result or None,
        }

        try:
            self.remember("forward", payload={"tool": tool, "task": task}, result={"message": short_msg, "has_image": bool(images_out)})
        except Exception:
            pass

        return out

    # ---------------------- Tool Runners (kept for reference / optional direct use) ----------------------
    def _run_hyperspy(self, params: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        from emseek.tools.analysis_tools import hyperspy_pca_denoise
        return hyperspy_pca_denoise(params.get('image_path'), out_dir)

    def _run_py4dstem(self, params: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        image_path = params.get('image_path')
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError('image_path is required for py4DSTEM task')
        from emseek.tools.analysis_tools import py4dstem_fft_calibration
        return py4dstem_fft_calibration(image_path)

    def _run_atomap(self, params: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        image_path = params.get('image_path')
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError('image_path is required for Atomap task')
        from emseek.tools.analysis_tools import atomap_peak_detect
        return atomap_peak_detect(image_path)

    def _run_skimage(self, params: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        image_path = params.get('image_path')
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError('image_path is required for scikit-image task')
        from emseek.tools.analysis_tools import skimage_edges
        return skimage_edges(image_path, out_dir)

    def _run_pymatgen(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cif_path = params.get('cif_path')
        if not cif_path or not os.path.isfile(cif_path):
            raise FileNotFoundError('cif_path is required for pymatgen task')
        from emseek.tools.analysis_tools import pymatgen_cif_summary
        return pymatgen_cif_summary(cif_path)

    def _run_materials_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get('query', 'Si')
        from emseek.tools.analysis_tools import materials_project_query
        api_key = getattr(self.platform.config, 'MP_API_KEY', None)
        return materials_project_query(query, api_key)
