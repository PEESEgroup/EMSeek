"""Generic input normalization and parsing utilities shared by agents.

These are extracted from agents to reduce duplication without changing behavior.
"""
from typing import Any, Dict


def sanitize_json_like(s: str) -> str:
    """
    Best-effort: convert pseudo-JSON into valid JSON.
    (Exact logic extracted from MaestroAgent._sanitize_json_like)
    """
    if not isinstance(s, str):
        return s  # type: ignore[return-value]
    t = s.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    t = t.strip("`")
    t = (
        t.replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'")
    )
    if t.startswith("{") and t.endswith("}"):
        if ('"' not in t) and ("'" in t):
            t = t.replace("'", '"')
    return t


def flatten_inputs_obj(obj: dict) -> dict:
    """Flatten extras into top-level and enforce a flat schema.
    (Exact logic extracted from MaestroAgent._flatten_inputs_obj)
    """
    if not isinstance(obj, dict):
        return {"query": str(obj)}
    out: Dict[str, Any] = dict(obj)
    extras = out.pop("extras", None)
    if isinstance(extras, dict):
        for k, v in extras.items():
            out.setdefault(k, v)
    for k in ("images", "cifs"):
        if k in out and out[k] in ("", [], {}):
            out.pop(k, None)
    return out


def is_dataurl_or_b64(s: str) -> bool:
    """Heuristic check used by SegMentor.
    (Exact logic extracted from SegMentorAgent._is_dataurl_or_b64)
    """
    if not isinstance(s, str) or not s:
        return False
    if s.strip().startswith("data:image/"):
        return True
    # loose test for base64 (common '=' padding or startswith '/9j' for jpg, 'iVBOR' for PNG)
    head = s.strip()[:16]
    return ("/9j" in head) or ("iVBOR" in head) or ("=" in s[-3:])

