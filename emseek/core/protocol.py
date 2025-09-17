import base64
import io
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union


# -------- Unified multimodal protocol helpers -------- #

ImageInput = Union[str, Dict[str, Any]]  # base64 string, data URL, or dict
AgentPayload = Dict[str, Any]


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:") and "," in s


def is_base64_str(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    # Common non-base64 prefixes (paths/URLs)
    sl = s.strip()
    lower = sl.lower()
    if sl.startswith(('/', './', '../')) or lower.startswith(('http://', 'https://')) or sl.startswith('/api/'):
        return False
    # If looks like a filename with image extension, treat as path
    if lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp')):
        return False
    # Accept proper data URLs immediately
    if is_data_url(sl):
        return True
    # Heuristic: must be reasonably long and mostly base64 alphabet
    try:
        sample = sl.split(',', 1)[-1]
        if len(sample) < 12:
            return False
        base64.b64decode(sample[:16] + '==')
        # Require only b64-safe chars for the sample
        for ch in sample[:16]:
            if not (ch.isalnum() or ch in '+/=_-'):
                return False
        return True
    except Exception:
        return False


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_base64_image(data_b64: str, out_dir: str, stem: str = "upload") -> str:
    ensure_dir(out_dir)
    if "," in data_b64:  # data URL
        data_b64 = data_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(data_b64)
    out_path = os.path.join(out_dir, f"{stem}_{int(time.time()*1000)}.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path


def path_to_base64_image(path: str) -> Optional[str]:
    """Read an image file from `path` and return base64 string (no data URL prefix).
    Returns None if path is invalid or read fails."""
    try:
        if not path or (not os.path.isfile(path)):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def to_image_refs(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize various agent return image forms to a list of typed image references.
    Enforces base64 for all interactive images when possible:
    - path string -> read and return {kind: 'base64', data: base64}
    - base64 string or data URL -> {kind: 'base64', data: '...'}
    - list like [[base64_or_path, caption], ...] -> list of dicts with caption (converted to base64)
    - list of refs [{'kind': 'path'|'base64', ...}] -> converted to base64 where needed
    - None -> []
    """
    def _ref_from_path(p: str, caption: Optional[str] = None) -> Dict[str, Any]:
        b64 = path_to_base64_image(p)
        if b64:
            return {"kind": "base64", "data": b64, **({"caption": caption} if caption else {})}
        # Fallback if read fails
        return {"kind": "path", "path": p, **({"caption": caption} if caption else {})}

    if obj is None:
        return []

    # Single string: base64/data URL or path
    if isinstance(obj, str):
        if is_base64_str(obj):
            return [{"kind": "base64", "data": obj}]
        return [_ref_from_path(obj)]

    # List like [[data, caption], ...]
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        refs: List[Dict[str, Any]] = []
        for item in obj:
            if not item:
                continue
            data = item[0]
            caption = item[1] if len(item) > 1 else None
            if isinstance(data, str) and is_base64_str(data):
                refs.append({"kind": "base64", "data": data, **({"caption": caption} if caption else {})})
            elif isinstance(data, str):
                refs.append(_ref_from_path(data, caption))
        return refs

    # List of plain strings: treat each as base64/data URL when applicable; otherwise as file paths
    # Previously we always treated plain strings as paths, which caused base64 strings inside lists
    # (e.g., images: ["iVBORw0K..."]) to be mislabeled as {kind: 'path', path: '<b64>'}.
    # That rendered as a text note instead of an actual image in the client.
    if isinstance(obj, list) and obj and isinstance(obj[0], str):
        refs: List[Dict[str, Any]] = []
        for s in obj:
            if not isinstance(s, str):
                continue
            s_clean = s.strip()
            if not s_clean:
                continue
            if is_base64_str(s_clean):
                refs.append({"kind": "base64", "data": s_clean})
            else:
                refs.append(_ref_from_path(s_clean))
        return refs

    # Already a list of refs? Convert 'path' kinds to base64 if possible
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "kind" in obj[0]:
        out: List[Dict[str, Any]] = []
        for ref in obj:
            try:
                if not isinstance(ref, dict):
                    continue
                k = ref.get("kind")
                if k == "base64" and ref.get("data"):
                    out.append(ref)
                elif k == "path" and ref.get("path"):
                    out.append(_ref_from_path(ref.get("path"), ref.get("caption")))
            except Exception:
                continue
        return out

    return []


# -------- Unified Agent Input (text/images/cifs/bbox) -------- #


def _as_list_paths(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else None
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out or None
    return None


def normalize_agent_input(raw: Any) -> AgentPayload:
    """Normalize arbitrary agent input into a unified dict with flat keys:
    {
      'text': str|None,
      'images': [file_path]|None,
      'cifs': [file_path]|None,
      'bbox': {x,y,w,h}|None,
      ... any additional keys passed through directly (no 'extras') ...
    }

    Backward-compatible mappings:
      - query/message/prompt -> text
      - image/image_path/image_paths -> images
      - cif/cif_path/cif_paths -> cifs
      - rect -> bbox
    Any other keys are preserved at the top level (no 'extras' nesting).
    """
    # Coerce to dict
    if isinstance(raw, dict):
        obj: Dict[str, Any] = dict(raw)
    else:
        s = raw if isinstance(raw, str) else str(raw)
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict):
                obj = {"text": s}
        except Exception:
            obj = {"text": s}

    # Extract canonical fields
    text = obj.get("text") or obj.get("query") or obj.get("message") or obj.get("prompt")

    # images: prefer explicit 'images' then fallbacks
    images = None
    for k in ("images", "image_paths"):
        images = _as_list_paths(obj.get(k))
        if images:
            break
    if images is None:
        # Single image path under common keys
        single_img = obj.get("image_path") or obj.get("image") or obj.get("path")
        images = _as_list_paths(single_img)

    # cifs: list of CIF file paths
    cifs = None
    for k in ("cifs", "cif_paths"):
        cifs = _as_list_paths(obj.get(k))
        if cifs:
            break
    if cifs is None:
        single_cif = obj.get("cif_path") or obj.get("cif")
        cifs = _as_list_paths(single_cif)

    # bbox
    bbox = obj.get("bbox") or obj.get("rect")
    if bbox is not None and not isinstance(bbox, dict):
        try:
            bbox = dict(bbox)  # best-effort
        except Exception:
            bbox = None

    # Other keys we didn't consume: keep at top-level (no 'extras')
    consumed = {"text", "query", "message", "prompt",
                "images", "image_paths", "image_path", "image", "path",
                "cifs", "cif_paths", "cif_path", "cif",
                "bbox", "rect"}
    passthrough = {k: v for k, v in obj.items() if k not in consumed}

    out: Dict[str, Any] = {
        "text": text if isinstance(text, str) else (str(text) if text is not None else None),
        "images": images,
        "cifs": cifs,
        "bbox": bbox,
    }
    # Merge passthrough keys directly (new info as keys)
    out.update(passthrough)
    # Remove keys with None to keep payload compact
    cleaned = {k: v for k, v in out.items() if v is not None}
    return cleaned


def build_stream_step(source: str, text: str = "", images: Any = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "type": "step",
        "time": _now_ts(),
        "source": source,
        "text": text or "",
        "images": to_image_refs(images),
        "data": data or None,
    }


def build_stream_final(text: str, user_id: Optional[str], images: Any = None, artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "type": "final",
        "time": _now_ts(),
        "text": text or "",
        "images": to_image_refs(images),
        "user_id": user_id,
        "artifacts": artifacts or None,
    }


def normalize_mm_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a multimodal request payload.

    Accepts variants like:
    - {text, image, bbox, options, agent, mode}
    - {inputs: {text, image, images:[], bbox}, metadata: {agent, options}}
    - image may be base64 string/data URL or {base64|path, bbox, model}
    """
    if not isinstance(payload, dict):
        return {"text": str(payload)}

    inputs = payload.get("inputs", payload)
    meta = payload.get("metadata", payload)
    text = inputs.get("text") if isinstance(inputs, dict) else None
    image = inputs.get("image") if isinstance(inputs, dict) else None
    images = inputs.get("images") if isinstance(inputs, dict) else None
    bbox = inputs.get("bbox") if isinstance(inputs, dict) else payload.get("bbox")
    options = meta.get("options") if isinstance(meta, dict) else payload.get("options")
    agent = meta.get("agent") if isinstance(meta, dict) else payload.get("agent")
    mode = meta.get("mode") if isinstance(meta, dict) else payload.get("mode")
    user_id = payload.get("user_id")

    # Allow image-level options override (Seg: model, bbox)
    if isinstance(image, dict):
        bbox = image.get("bbox", bbox)
        if options is None and any(k in image for k in ("model", "elements", "top_k")):
            # Map common fields into options namespace
            options = {k: v for k, v in image.items() if k in ("model", "elements", "top_k")}

    return {
        "text": text,
        "image": image,
        "images": images,
        "bbox": bbox,
        "options": options or {},
        "agent": agent,          # preferred agent or None
        "mode": mode or "auto",  # 'auto' (EMSeek decides) or 'direct'
        "user_id": user_id,
    }
