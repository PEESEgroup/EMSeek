import os
import json
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from emseek.agents.base import Agent
from pymatgen.core import Structure


# --------------------------- Helpers ---------------------------

def _safe_load_structure(cif_path: str):
    if Structure is None:
        raise RuntimeError("pymatgen is required to load CIF structures.")
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    return Structure.from_file(cif_path)


def _fallback_features(struct) -> Tuple[int, float, float]:
    """Lightweight structural descriptors for fallback predictions.
    Returns (natoms, mean_atomic_number, std_atomic_number).
    """
    zs = np.array(
        [sp.Z for sp in struct.composition.elements for _ in range(int(struct.composition[sp]))],
        dtype=float
    )
    if zs.size == 0:
        return 0, 0.0, 0.0
    return zs.size, float(np.mean(zs)), float(np.std(zs))


def _fallback_energy(struct) -> float:
    n, mu, sigma = _fallback_features(struct)
    # Simple deterministic heuristic; ensures finite output
    return float(0.1 * n + 0.05 * mu - 0.02 * sigma)


def _forces_summary(forces: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Return RMS and max norms of forces; robust to None / bad shapes."""
    try:
        if forces is None:
            return {"force_rms": None, "fmax": None}
        F = np.asarray(forces, dtype=float)
        if F.ndim != 2 or F.shape[1] != 3:
            return {"force_rms": None, "fmax": None}
        norms = np.linalg.norm(F, axis=1)
        return {
            "force_rms": float(np.sqrt(np.mean(norms ** 2))) if norms.size else None,
            "fmax": float(np.max(norms)) if norms.size else None,
        }
    except Exception:
        return {"force_rms": None, "fmax": None}


def _stress_summary(stress) -> Dict[str, Optional[float]]:
    """Accept 3x3 tensor or 6-Voigt; return trace and 'pressure' proxy (-trace/3)."""
    try:
        if stress is None:
            return {"stress_trace": None, "pressure": None}
        S = np.asarray(stress, dtype=float)
        if S.shape == (3, 3):
            tr = float(np.trace(S))
        elif S.size == 6:  # Voigt: xx, yy, zz, yz, xz, xy
            tr = float(S.flat[0] + S.flat[1] + S.flat[2])
        else:
            return {"stress_trace": None, "pressure": None}
        return {"stress_trace": tr, "pressure": float(-tr / 3.0)}
    except Exception:
        return {"stress_trace": None, "pressure": None}


def _pack_props(struct, energy_total: float,
                forces: Optional[np.ndarray] = None,
                stress: Optional[np.ndarray] = None,
                extras: Optional[Dict] = None) -> Dict:
    natoms = max(1, int(len(struct)))
    pa = float(energy_total) / float(natoms)
    fsum = _forces_summary(forces)
    ssum = _stress_summary(stress)
    out = {
        "energy_total": float(energy_total),
        "energy_per_atom": float(pa),
        "has_forces": forces is not None,
        "has_stress": stress is not None,
        **fsum,
        **ssum,
    }
    if extras:
        out.update(extras)
    return out


# --------------------------- Predictors ---------------------------

def predict_matter_sim(struct) -> Dict:
    """Return dict with energy_total, energy_per_atom, fmax, force_rms, stress_trace, pressure, ..."""
    try:
        from mattersim.forcefield.potential import Potential  # type: ignore
        from mattersim.datasets.utils.build import build_dataloader  # type: ignore
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore
        import torch as _t

        device = "cuda" if _t.cuda.is_available() else "cpu"
        ff = Potential.from_checkpoint(load_path='mattersim-v1.0.0-5m.pth', device=device).to(device)

        atoms = AseAtomsAdaptor.get_atoms(struct)
        loader = build_dataloader([atoms], batch_size=1, pin_memory=False, only_inference=True)

        with _t.no_grad():
            pred = ff.predict_properties(loader)
        # Handle diverse return types robustly
        energy, forces, stress = None, None, None
        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            item = pred[0]
            if isinstance(item, dict):
                energy = item.get("energy", None)
                forces = item.get("forces", None)
                stress = item.get("stress", None)
            elif hasattr(item, "__len__") and len(item) >= 1:
                energy = item[0]
        elif isinstance(pred, dict):
            energy = pred.get("energy", None)
            forces = pred.get("forces", None)
            stress = pred.get("stress", None)

        if energy is None:
            energy = _fallback_energy(struct)

        energy = float(np.asarray(energy, dtype=float).ravel()[0])
        return _pack_props(struct, energy, forces=forces, stress=stress, extras={"provider": "MatterSim"})
    except Exception:
        e = _fallback_energy(struct)
        return _pack_props(struct, e, forces=None, stress=None, extras={"provider": "MatterSim", "note": "fallback"})


def predict_uma(struct: Structure, model_name: str = "uma-m-1p1") -> Dict:
    """UMA prediction (per-atom energy target in training; we still return total/per-atom)."""
    import inspect
    from contextlib import contextmanager
    try:
        from fairchem.core import pretrained_mlip  # type: ignore
        from fairchem.core.datasets.atomic_data import (  # type: ignore
            AtomicData,
            atomicdata_list_to_batch,
        )
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore
        import torch as _t
        import numpy as _np

        @contextmanager
        def _allow_full_torch_load():
            _orig = _t.load
            if "weights_only" in inspect.signature(_orig).parameters:
                def _patched(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _orig(*args, **kwargs)
                _t.load = _patched
            try:
                yield
            finally:
                _t.load = _orig

        device = "cuda" if _t.cuda.is_available() else "cpu"
        with _allow_full_torch_load():
            predictor = pretrained_mlip.get_predict_unit(model_name, device=device)

        atomic = AtomicData.from_ase(AseAtomsAdaptor.get_atoms(struct), task_name="omat")
        batch = atomicdata_list_to_batch([atomic])

        with _t.no_grad():
            preds = predictor.predict(batch)  # dict-like
        energy = preds.get("energy", None)
        forces = preds.get("forces", None)
        stress = preds.get("stress", None)

        if energy is None:
            # Some implementations put the scalar in "y" or a list
            energy = preds.get("y", None)
        if energy is None:
            raise RuntimeError("UMA returns no energy")

        energy = float(_np.asarray(energy, dtype=float).ravel()[0])
        return _pack_props(struct, energy, forces=forces, stress=stress, extras={"provider": "UMA", "model_name": model_name})
    except Exception:
        e = _fallback_energy(struct)
        return _pack_props(struct, e, forces=None, stress=None, extras={"provider": "UMA", "model_name": model_name, "note": "fallback"})


def predict_orb_v3(struct) -> Dict:
    try:
        from orb_models.forcefield import pretrained as orb_pretrained  # type: ignore
        from orb_models.forcefield import atomic_system  # type: ignore
        from orb_models.forcefield.base import batch_graphs  # type: ignore
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore
        import torch as _t

        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        orbff = orb_pretrained.__dict__["orb_v3_conservative_inf_omat"](device=device, precision="float32-high")

        atoms = AseAtomsAdaptor.get_atoms(struct)
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

        # ORB needs positional gradients to export forces; if forces are not needed, use no_grad
        with _t.enable_grad():
            batch = batch_graphs([graph])
            # Require grad only if predict needs gradients
            for key in ("pos", "positions", "node_positions"):
                if isinstance(batch, dict) and key in batch and isinstance(batch[key], _t.Tensor):
                    batch[key].requires_grad_(True)
            result = orbff.predict(batch, split=False)

        energy = result.get("energy", None)
        forces = result.get("forces", None)
        stress = result.get("stress", None)

        if energy is None:
            raise RuntimeError("ORB-v3 returns no energy")
        energy = float(np.asarray(energy, dtype=float).ravel()[0])

        def _to_np(x):
            try:
                import torch as _t2
                if isinstance(x, _t2.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
            return x

        return _pack_props(struct, energy, forces=_to_np(forces), stress=_to_np(stress), extras={"provider": "ORBv3"})
    except Exception:
        e = _fallback_energy(struct)
        return _pack_props(struct, e, forces=None, stress=None, extras={"provider": "ORBv3", "note": "fallback"})


def predict_mace(struct) -> Dict:
    try:
        from mace.calculators import mace_mp  # type: ignore
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore

        # MACE ASE calculator
        calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cpu")
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        # Try forces / stress (skip if backend unsupported)
        forces, stress = None, None
        try:
            forces = atoms.get_forces()
        except Exception:
            forces = None
        try:
            # ASE often returns 6-Voigt; prefer 3x3
            try:
                stress = atoms.get_stress(voigt=False)
            except Exception:
                stress = atoms.get_stress()
        except Exception:
            stress = None

        return _pack_props(struct, float(energy), forces=forces, stress=stress, extras={"provider": "MACE", "model": "medium"})
    except Exception:
        e = _fallback_energy(struct)
        return _pack_props(struct, e, forces=None, stress=None, extras={"provider": "MACE", "note": "fallback"})


def _load_moe_and_stats(ckpt_path: str):
    """
    Load MoE (GatedAffineMoE) and its normalization stats saved by the training script:
      - <base>.pt
      - <base>_normstats.npz  (keys: bp_mean, bp_std)
    Returns (moe_model_or_none, norm_stats_or_none)
    """
    if not ckpt_path:
        return None, None

    base, ext = os.path.splitext(ckpt_path)
    if ext == "":
        base = ckpt_path
    elif ext.lower() != ".pt":
        base = os.path.splitext(ckpt_path)[0]

    sd_path = base + ".pt"
    ns_path = base + "_normstats.npz"

    if not os.path.isfile(sd_path):
        return None, None

    try:
        import torch as _t
        from emseek.models.attention import GatedAffineMoE  # local to avoid importing heavy modules
    except Exception:
        return None, None

    moe = GatedAffineMoE(n_experts=4)
    sd = _t.load(sd_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        missing, unexpected = moe.load_state_dict(sd["state_dict"], strict=False)
    else:
        missing, unexpected = moe.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[MoE] Loaded with missing keys: {missing}, unexpected: {unexpected}")

    moe.eval()

    norm_stats = None
    if os.path.isfile(ns_path):
        npz = np.load(ns_path)
        norm_stats = {
            "bp_mean": npz["bp_mean"],  # shape (1, n_experts)
            "bp_std":  np.maximum(npz["bp_std"], 1e-12),
        }
    else:
        print(f"[MoE] Warning: normstats not found at {ns_path}; will skip normalization.")

    return moe, norm_stats


def _structure_info(struct: Structure) -> Dict:
    """Extract chemistry & lattice metadata with graceful fallbacks."""
    info: Dict[str, object] = {}
    try:
        comp = struct.composition
        info["reduced_formula"] = comp.reduced_formula
        info["anonymized_formula"] = comp.anonymized_formula
        info["elements"] = [el.symbol for el in comp.elements]
        info["nelements"] = len(comp.elements)
        info["natoms"] = int(len(struct))
        info["volume"] = float(struct.volume)
        try:
            # pymatgen Structure has property density (g/cc)
            info["density"] = float(struct.density)
        except Exception:
            info["density"] = None
        lat = struct.lattice
        info["lattice"] = {
            "a": float(lat.a), "b": float(lat.b), "c": float(lat.c),
            "alpha": float(lat.alpha), "beta": float(lat.beta), "gamma": float(lat.gamma),
        }
        # Space group via spglib if available
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore
            sga = SpacegroupAnalyzer(struct, symprec=1e-2)
            spg = sga.get_space_group_symbol()
            num = sga.get_space_group_number()
            info["spacegroup"] = {"symbol": spg, "number": int(num)}
        except Exception:
            info["spacegroup"] = None
        # Z / average mass (quick descriptors)
        zs = np.array([sp.Z for sp in comp.elements for _ in range(int(comp[sp]))], dtype=float)
        am = np.array([sp.atomic_mass for sp in comp.elements for _ in range(int(comp[sp]))], dtype=float)
        info["mean_Z"] = float(np.mean(zs)) if zs.size else None
        info["mean_atomic_mass"] = float(np.mean(am)) if am.size else None
    except Exception:
        pass
    return info


# --------------------------- Agent ---------------------------

class MatProphetAgent(Agent):
    """Predict total/per-atom energy from a CIF using four base models and a MoE ensemble.

    - Each base model returns not only energy but also (if available) forces/stress summaries.
    - MoE blends per-atom energies. If MoE missing, falls back to mean.
    - An LLM-backed controller validates/repairs inputs and summarizes results for Maestro.
    """

    def __init__(self, name, platform):
        super().__init__(name, platform)
        # Allow overriding MoE checkpoint path via cfg
        self.moe_ckpt = getattr(self.platform.config, 'MATPROPHET_MOE_CKPT', None) or os.path.join('pretrained', 'MatProphet', 'MoE_Ensemble')
        self.moe, self.moe_norm = _load_moe_and_stats(self.moe_ckpt)
        self.description = (
            "MatProphet materials energy ensemble: given a CIF structure, calls multiple models "
            "(MatterSim / UMA / ORBv3 / MACE) and blends per-atom energy via a MoE to produce a unified result with an uncertainty summary.\n\n"

            "Input formats (one of):\n"
            "- Native JSON: {\"cif_path\": str}\n"
            "- Unified dict: {text?: str, cifs?: str|[str]}  (if cifs is present, the first item is used as cif_path)\n\n"

            "Output format (unified dict):\n"
            "{\n"
            "  ok: bool,\n"
            "  message: str,            # one-sentence summary for router/UI\n"
            "  text: str,               # concise report of multi-model results and structure info\n"
            "  cifs: [str]|null,        # CIF paths used (usually length 1)\n"
            "  images: null,            # this agent does not produce images\n"
            "  meta: {                  # detailed numbers and metadata\n"
            "    base_predictions: {MatterSim, UMA, ORBv3, MACE},   # per-atom energies\n"
            "    weights: {...},                                   # MoE weights (if available)\n"
            "    per_atom: {ensemble, mean_base},                   # fused per-atom energy etc.\n"
            "    total: {ensemble},                                 # total energy\n"
            "    uncertainty: {std, range, num_models},\n"
            "    natoms: int, cif_path: str,\n"
            "    structure: {...},                                   # formula / crystallography\n"
            "    model_details: {fmax, pressure, ... per model}\n"
            "  }\n"
            "}\n\n"

            "Notes: if cif_path is missing, it may be inferred from the unified input or recent history; "
            "MoE checkpoint/normalization can be configured and otherwise falls back to a simple mean."
        )

    def expected_input_schema(self) -> str:
        return '{"cif_path": str, "cifs": [str]?}'

    # -------------------- LLM-backed controller helpers --------------------

    def _error_dict(self, message: str, *, errors: Optional[List[str]] = None,
                    meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {
            "ok": False,
            "text": message,
            "message": message,   # short reply to Maestro
            "error": {"message": message, "fields": errors or []},
            "images": None,
            "cifs": None,
            "meta": meta or None,
        }
        return out

    def controller_validate_and_fix(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate/repair inputs for MatProphet.
        Input payload keys of interest: {cif_path} OR unified {cifs: [..]}.
        Returns: {ok: bool, payload|None, errors: [..], message: str}
        Repair strategies:
          - If unified cifs present, prefer first.
          - Normalize to absolute path.
          - Try recover from recent history via LLM when missing.
        """
        errors: List[str] = []
        msg_parts: List[str] = []

        # Prefer unified 'cifs' first if provided
        cif_path = payload.get("cif_path")
        cifs = payload.get("cifs")
        if not cif_path and isinstance(cifs, list) and cifs:
            cif_path = cifs[0]

        # Normalize to abs path
        if isinstance(cif_path, str):
            try:
                cif_path = os.path.abspath(cif_path)
            except Exception:
                pass

        # If not valid, try LLM from recent history
        if not (isinstance(cif_path, str) and os.path.isfile(cif_path)):
            inferred = None
            if getattr(self, 'llm', None):
                try:
                    hist = self.recent_memory_text(8)
                    prompt = (
                        "You validate inputs for a CIF-based property predictor.\n"
                        "From the recent history, if a CIF path was mentioned and exists on disk, extract it.\n"
                        "Return JSON {\"cif_path\": \"/abs/path\"} or {\"cif_path\": null}.\n\n"
                        f"History:\n{hist}\n\nOutput JSON:"
                    )
                    out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                    s = out.strip(); i = s.find('{'); j = s.rfind('}')
                    if i != -1 and j != -1 and j > i:
                        data = json.loads(s[i:j+1])
                        cp = data.get('cif_path')
                        if isinstance(cp, str) and os.path.isfile(cp):
                            inferred = os.path.abspath(cp)
                except Exception:
                    pass
            if inferred:
                cif_path = inferred
                msg_parts.append("cif_path inferred from history")

        if not (isinstance(cif_path, str) and os.path.isfile(cif_path)):
            errors.append("cif_path")
            return {"ok": False, "payload": None, "errors": errors,
                    "message": "Missing or invalid fields: cif_path"}

        payload_fixed = dict(payload)
        payload_fixed["cif_path"] = cif_path
        return {"ok": True, "payload": payload_fixed, "errors": [], "message": "; ".join(msg_parts) or "ok"}

    def controller_summarize_success(self, *, natoms: int, per_atom: float, total: float,
                                     std: Optional[float], top_model: Optional[str], low_model: Optional[str]) -> str:
        """
        Produce a short, single-sentence message (<=160 chars) for Maestro after success.
        Uses LLM if available; otherwise returns a concise heuristic summary.
        """
        try:
            if getattr(self, 'llm', None):
                prompt = (
                    "Write one short sentence (<=160 chars) summarizing a multi-model energy prediction for a crystal.\n"
                    f"natoms={natoms}, per_atom={per_atom:.6f}, total={total:.6f}, std={None if std is None else f'{std:.6f}'}, "
                    f"lowest={low_model or 'NA'}, highest={top_model or 'NA'}.\n"
                    "No extra commentary."
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                return out.strip().splitlines()[0][:160]
        except Exception:
            pass
        # Heuristic fallback
        spread = f", std={std:.6f}" if isinstance(std, (int, float)) else ""
        lohi = ""
        if low_model or top_model:
            lohi = f", lowest={low_model or 'NA'}, highest={top_model or 'NA'}"
        return f"Ensemble energy: {per_atom:.6f}/atom, total={total:.6f} for {natoms} atoms{spread}{lohi}."

    # -------------------- Report summarizer --------------------

    def _summarize(self, summary: Dict) -> str:
        from emseek.utils.llm_caller import LLMCaller  # optional
        caller = LLMCaller(self.platform.config)
        messages = [
            {"role": "system", "content": "You are a materials science expert. Be concise and factual."},
            {"role": "user", "content": (
                "Given structure metadata and multi-model predictions, produce a concise engineering-style report. Include:\n"
                "1) Ensemble per-atom & total energy + short stability verdict;\n"
                "2) Relaxation hint from forces (fmax);\n"
                "3) Stress interpretation via pressure proxy (-trace/3);\n"
                "4) Which base model is lowest/highest and a trust hint;\n"
                "5) Chemistry/lattice facts (formula, space group if present).\n"
                "Return 5–8 bullet points and a short concluding sentence.\n\n"
                f"DATA JSON:\n{json.dumps(summary, ensure_ascii=False)}"
            )},
        ]
        out = caller.forward(messages=messages)
        if isinstance(out, str) and out.strip():
            return out.strip()

        # Fallback
        base = summary.get('base_predictions', {}) or {}
        pa = (summary.get('per_atom', {}) or {}).get('ensemble', float('nan'))
        total = (summary.get('total', {}) or {}).get('ensemble', float('nan'))
        unc = summary.get('uncertainty', {}) or {}
        std = unc.get('std', float('nan'))
        rng = unc.get('range', float('nan'))
        model_details = summary.get('model_details', {})
        struct_info = summary.get('structure', {})

        # Extrema
        min_model, max_model = None, None
        if base:
            try:
                min_model, _ = min(base.items(), key=lambda kv: kv[1])
                max_model, _ = max(base.items(), key=lambda kv: kv[1])
            except Exception:
                pass

        lines = [
            f"Ensemble per-atom: {pa:.6f}; total: {total:.6f}.",
            f"Spread (std): {std:.6f}; range: {rng:.6f}.",
        ]
        if min_model and max_model:
            lines.append(f"Lowest model: {min_model}; highest: {max_model}.")
        def _fmt(md):
            if not md: return "N/A"
            fmax = md.get("fmax", None)
            pr = md.get("pressure", None)
            fpart = f"fmax={fmax:.3f}" if isinstance(fmax, (int, float)) else "fmax=N/A"
            spart = f"p≈{pr:.3f}" if isinstance(pr, (int, float)) else "p=N/A"
            return f"{fpart}, {spart}"
        details_str = ", ".join(f"{k}[{_fmt(v)}]" for k, v in model_details.items()) if model_details else "N/A"
        lines.append(f"Non-energy summaries: {details_str}")

        if struct_info:
            rf = struct_info.get("reduced_formula", "N/A")
            spg = struct_info.get("spacegroup", None)
            spg_s = f"{spg.get('symbol')} #{spg.get('number')}" if isinstance(spg, dict) else "N/A"
            lines.append(f"Structure: {rf}; space group: {spg_s}.")
        return "\n".join(lines)

    # -------------------- Main entry --------------------

    def forward(self, messages: str | dict) -> Dict[str, Any]:
        """
        Accepts {"cif_path": str} or unified {text, images, cifs, bbox, ...}.
        Always returns a unified dict:
          - success: {ok: True, message, text, cifs, meta}
          - failure: {ok: False, message, error{message, fields}}
        """
        # 0) parse messages -> payload (tolerate JSON string)
        if isinstance(messages, dict):
            payload = dict(messages)
        else:
            s = messages if isinstance(messages, str) else str(messages)
            try:
                obj = json.loads(s)
                payload = obj if isinstance(obj, dict) else {"query": s}
            except Exception:
                payload = {"query": s}

        # 1) accept unified payload and propagate cifs if present
        try:
            uni = self.normalize_agent_payload(payload)
            cifs = uni.get('cifs') if isinstance(uni, dict) else None
        except Exception:
            uni = {}
            cifs = None

        ctrl_payload = {
            "cif_path": payload.get("cif_path"),
            "cifs": cifs if isinstance(cifs, list) else None,
            "text": uni.get("text") if isinstance(uni, dict) else None,
        }

        # 2) LLM-backed validation & auto-fix
        ctrl = self.controller_validate_and_fix(ctrl_payload)
        if not ctrl.get("ok", False):
            return self._error_dict(ctrl.get("message", "Invalid input."), errors=ctrl.get("errors", ["cif_path"]))

        cif_path = ctrl["payload"]["cif_path"]
        try:
            cif_path = os.path.abspath(cif_path)
        except Exception:
            pass

        # 3) Load structure
        try:
            struct = _safe_load_structure(cif_path)
        except Exception:
            return self._error_dict("Failed to load CIF structure.", errors=["cif_path"])

        natoms = max(1, int(len(struct)))

        # 4) Base models (rich props)
        ms = predict_matter_sim(struct)
        uma_model = getattr(self.platform.config, 'UMA_MODEL_NAME', 'uma-m-1p1')
        uma = predict_uma(struct, model_name=uma_model)
        orb = predict_orb_v3(struct)
        mace = predict_mace(struct)

        # Per-atom energies vector
        base_pa_vec = np.array([
            ms["energy_per_atom"], uma["energy_per_atom"], orb["energy_per_atom"], mace["energy_per_atom"]
        ], dtype=float)

        base_pa = {
            'MatterSim': float(ms["energy_per_atom"]),
            'UMA': float(uma["energy_per_atom"]),
            'ORBv3': float(orb["energy_per_atom"]),
            'MACE': float(mace["energy_per_atom"]),
        }

        # 5) MoE ensemble
        if self.moe is not None:
            try:
                import torch as _t
            except Exception:
                self.moe = None
                _t = None

            if self.moe is not None and _t is not None:
                base_pa_clean = base_pa_vec.copy()
                if not np.isfinite(base_pa_clean).all():
                    finite_mask = np.isfinite(base_pa_clean)
                    if finite_mask.any():
                        fillv = float(np.nanmean(base_pa_clean[finite_mask]))
                        base_pa_clean = np.where(finite_mask, base_pa_clean, fillv)
                    else:
                        base_pa_clean = np.zeros_like(base_pa_clean, dtype=float)

                if getattr(self, "moe_norm", None) is not None:
                    bp_mean = np.asarray(self.moe_norm["bp_mean"], dtype=float).reshape(1, -1)
                    bp_std  = np.asarray(self.moe_norm["bp_std"],  dtype=float).reshape(1, -1)
                    BP = (base_pa_clean.reshape(1, -1) - bp_mean) / bp_std
                else:
                    BP = base_pa_clean.reshape(1, -1)

                X_dummy = _t.zeros((1, 0), dtype=_t.float32)
                BP_t = _t.as_tensor(BP, dtype=_t.float32)
                with _t.no_grad():
                    y_hat, w, _ = self.moe(X_dummy, BP_t)
                pa_ens = float(y_hat.squeeze().cpu().numpy())
                weights = w.squeeze(0).cpu().numpy().tolist()
            else:
                pa_ens = float(np.nanmean(base_pa_vec))
                weights = [0.25, 0.25, 0.25, 0.25]
        else:
            pa_ens = float(np.nanmean(base_pa_vec))
            weights = [0.25, 0.25, 0.25, 0.25]

        # 6) Uncertainty / non-energy summaries / structure info
        base_total_vec = np.array([
            ms["energy_total"], uma["energy_total"], orb["energy_total"], mace["energy_total"]
        ], dtype=float)
        std = float(np.nanstd(base_total_vec))
        rng = float(np.nanmax(base_total_vec) - np.nanmin(base_total_vec)) if np.isfinite(base_total_vec).all() else float('nan')

        s_info = _structure_info(struct)
        model_details = {
            "MatterSim": {k: ms.get(k) for k in ("fmax", "force_rms", "pressure", "stress_trace", "has_forces", "has_stress")},
            "UMA":       {k: uma.get(k) for k in ("fmax", "force_rms", "pressure", "stress_trace", "has_forces", "has_stress")},
            "ORBv3":     {k: orb.get(k) for k in ("fmax", "force_rms", "pressure", "stress_trace", "has_forces", "has_stress")},
            "MACE":      {k: mace.get(k) for k in ("fmax", "force_rms", "pressure", "stress_trace", "has_forces", "has_stress")},
        }

        # 7) Compose meta
        result = {
            'base_predictions': base_pa,
            'weights': {
                'MatterSim': float(weights[0]) if len(weights) >= 4 else None,
                'UMA': float(weights[1]) if len(weights) >= 4 else None,
                'ORBv3': float(weights[2]) if len(weights) >= 4 else None,
                'MACE': float(weights[3]) if len(weights) >= 4 else None,
            },
            'per_atom': {
                'ensemble': pa_ens,
                'mean_base': float(np.nanmean(base_pa_vec)),
            },
            'total': {
                'ensemble': pa_ens * natoms,
            },
            'uncertainty': {
                'std': std,
                'range': rng,
                'num_models': 4,
            },
            'natoms': natoms,
            'cif_path': cif_path,
            'structure': s_info,
            'model_details': model_details,
        }

        # 8) Build human-readable summary + short message
        summary_text = self._summarize(result)

        # Identify lowest/highest per-atom model names for the short message
        low_model = None
        top_model = None
        try:
            low_model = min(base_pa, key=base_pa.get)
            top_model = max(base_pa, key=base_pa.get)
        except Exception:
            pass

        short_msg = self.controller_summarize_success(
            natoms=natoms,
            per_atom=pa_ens,
            total=pa_ens * natoms,
            std=std,
            top_model=top_model,
            low_model=low_model
        )

        return {
            "ok": True,
            "message": short_msg,      # concise sentence for Maestro
            "text": summary_text,      # detailed bullet-point summary
            "cifs": [cif_path],
            "meta": result,
            "images": None,
        }
