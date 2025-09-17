import os
import io
import json
import math
import base64
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    F = None      # type: ignore
    TORCH_OK = False

from emseek.agents.base import Agent
from emseek.retrieval.crystalforge import (
    load_gray, build_cif_index, parse_elements_text, filter_by_elements,
    score_candidates,
)

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


# ---------- Minimal copies from training/CrystalForge/train.py for inference-only ----------

def load_mono_as01(path: str, norm: str = 'percentile', p_lo: float = 1.0, p_hi: float = 99.0, auto_invert: bool = False):
    with Image.open(path) as img:
        if img.mode in ('LA', 'RGBA', 'RGB'):
            img = img.convert('L')
        arr = np.array(img, dtype=np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if norm == 'none':
        denom = 255.0 if vmax <= 255.0 else 65535.0
        x = arr / max(denom, 1e-8)
        x = np.clip(x, 0.0, 1.0)
    elif norm == 'minmax':
        if vmax <= vmin + 1e-8: x = np.zeros_like(arr, dtype=np.float32)
        else: x = (arr - vmin) / (vmax - vmin)
    else:  # percentile
        lo = np.percentile(arr, p_lo); hi = np.percentile(arr, p_hi)
        if hi <= lo + 1e-8: x = np.zeros_like(arr, dtype=np.float32)
        else: x = (arr - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
    if auto_invert and float(x.mean()) > 0.6:
        x = 1.0 - x
    if TORCH_OK:
        return torch.from_numpy(x).unsqueeze(0)
    else:
        return x[None, ...]


def overlay_on_gray(lr, mask, alpha: float = 0.5, color=(0, 255, 0)) -> Image.Image:
    if TORCH_OK and isinstance(lr, torch.Tensor):
        base = (lr.detach().cpu().squeeze(0).numpy() * 255.0).astype(np.uint8)
        m = (mask.detach().cpu().squeeze(0).numpy() > 0).astype(np.float32)[..., None]
    else:
        base = (np.squeeze(lr, 0) * 255.0).astype(np.uint8)
        m = (np.squeeze(mask, 0) > 0).astype(np.float32)[..., None]
    H, W = base.shape
    base_rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    # Only draw inside mask; keep zeros outside mask
    out = base_rgb * ((1.0 - alpha) * m) + color_arr * (alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='RGB')


def _sliding_window_predict(lr_img, model, tile: int = 512, stride: Optional[int] = None,
                            device: str = 'cuda' if TORCH_OK and torch and torch.cuda.is_available() else 'cpu', amp: bool = True,
                            tile_bs: int = 8, channels_last: bool = True):
    if not TORCH_OK:
        raise RuntimeError('sliding window prediction requires torch')
    if stride is None: stride = tile // 2
    _, H, W = lr_img.shape
    pad_h = (math.ceil((H - tile) / stride) * stride + tile - H) if H > tile else (tile - H)
    pad_w = (math.ceil((W - tile) / stride) * stride + tile - W) if W > tile else (tile - W)
    top = max(0, pad_h // 2); bottom = max(0, pad_h - top)
    left = max(0, pad_w // 2); right = max(0, pad_w - left)
    lr_pad = F.pad(lr_img.unsqueeze(0), (left, right, top, bottom), mode='reflect').squeeze(0)
    _, Hp, Wp = lr_pad.shape
    prob_acc = torch.zeros((1, Hp, Wp), dtype=torch.float32)
    cnt_acc  = torch.zeros((1, Hp, Wp), dtype=torch.float32)
    coords = [(y,x) for y in range(0, Hp - tile + 1, stride) for x in range(0, Wp - tile + 1, stride)]
    model.eval(); amp_en = (device.startswith('cuda') and amp)
    for i in range(0, len(coords), tile_bs):
        batch_coords = coords[i:i+tile_bs]
        patches = [lr_pad[:, y:y+tile, x:x+tile] for (y, x) in batch_coords]
        batch = torch.stack(patches, dim=0)
        if channels_last: batch = batch.contiguous(memory_format=torch.channels_last)
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_en), torch.inference_mode():
            logits = model(batch); probs = torch.sigmoid(logits)
        probs_cpu = probs.detach().cpu()
        for j, (y, x) in enumerate(batch_coords):
            prob_acc[:, y:y+tile, x:x+tile] += probs_cpu[j]
            cnt_acc[:,  y:y+tile, x:x+tile] += 1
        del batch, logits, probs, probs_cpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    probs_full = prob_acc / (cnt_acc + 1e-8)
    probs = probs_full[:, top:top+H, left:left+W]
    return probs.clamp(0,1)


def _prepare_model_from_ckpt(ckpt_path: str, device: str,
                             channels_last: bool = True):
    """Load UNet++ weights saved from training/CrystalForge/train.py.

    Training wraps SMP's UnetPlusPlus in a UNetPP class and saves state_dict
    with keys prefixed by 'net.'. Here we instantiate SMP's model directly and
    strip the 'net.' prefix (and common wrappers like 'module.').
    """
    import segmentation_models_pytorch as smp

    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None

    ck = torch.load(ckpt_path, map_location=device)
    cfgd = ck.get('cfg', {})
    encoder_name = cfgd.get('encoder_name', 'resnet18')
    encoder_weights = cfgd.get('encoder_weights', None)

    net = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)

    if channels_last:
        net = net.to(memory_format=torch.channels_last)

    sd = ck.get('model') or ck
    if not isinstance(sd, dict):
        return net.eval()

    # Remap keys: drop 'net.' or 'module.' or 'model.' prefixes
    remapped = {}
    for k, v in sd.items():
        nk = k
        for pref in ('net.', 'module.', 'model.'):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        remapped[nk] = v

    net.load_state_dict(remapped, strict=True)
    net.eval()
    return net


def _binarize_otsu(x01):
    # simple Otsu on numpy for fallback
    import numpy as _np
    from skimage.filters import threshold_otsu as _thr
    if TORCH_OK and isinstance(x01, torch.Tensor):
        x = x01.squeeze(0).numpy().astype(np.float32)
        t = _thr(x)
        return torch.from_numpy((x >= t).astype(np.float32)).unsqueeze(0)
    else:
        x = np.squeeze(x01, 0).astype(np.float32)
        t = _thr(x)
        return (x >= t).astype(np.float32)[None, ...]


def save_png01(path: str, ten):
    if TORCH_OK and isinstance(ten, torch.Tensor):
        ten = ten.detach().cpu().clamp(0,1)
        arr = (ten.squeeze(0).numpy() * 255.0 + 0.5).astype(np.uint8)
    else:
        arr = (np.squeeze(ten, 0).clip(0,1) * 255.0 + 0.5).astype(np.uint8)
    ensure_dir(os.path.dirname(path))
    Image.fromarray(arr, mode='L').save(path)


def encode_image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format='PNG'); return base64.b64encode(buf.getvalue()).decode('utf-8')


# def render_cif_snapshot(cif_path: str, out_path: str, size: Tuple[int, int] = (400, 400)) -> str:
#     from ase.io import read as ase_read
#     from ase.visualize.plot import plot_atoms
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     atoms = ase_read(cif_path)
#     fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
#     ax = fig.add_subplot(111)
#     plot_atoms(atoms, ax, radii=0.3, rotation=('45x,30y,0z'), show_unit_cell=2)
#     ax.set_axis_off()
#     ensure_dir(os.path.dirname(out_path))
#     fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     return out_path


def render_cif_snapshot(
    cif_path: str,
    out_path: str,
    size: tuple[int, int] = (1200, 1200),
    replicate: tuple[int, int, int] = (3, 3, 1),
    atom_radius: float = 0.10,      # Per-atom radius (highest priority)
    radius_scale: float = 1.0,      # Global scaling on top of per-atom radius
    bond_width: float = 0.20,       # Stick thickness in ball-and-stick
) -> str:
    """
    Render CIF to a high-quality image using OVITO (headless) with
    auto view orientation, replication, and ball-and-stick visualization.
    Returns out_path.
    """
    import os, math
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # required for headless environments

    import numpy as np
    from ovito.io import import_file
    from ovito.modifiers import (
        WrapPeriodicImagesModifier,
        ReplicateModifier,
        CreateBondsModifier,
        ComputePropertyModifier,
    )
    from ovito.vis import Viewport, TachyonRenderer, ParticlesVis

    def ensure_dir(p: str):
        if p and not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ------------ 1) Load + wrap + replicate + ball-and-stick ------------
    pipeline = import_file(cif_path)
    pipeline.modifiers.append(WrapPeriodicImagesModifier())
    pipeline.modifiers.append(ReplicateModifier(num_x=replicate[0], num_y=replicate[1], num_z=replicate[2]))

    # Configure the bond "stick"
    cb = CreateBondsModifier()
    cb.vis.width = bond_width
    pipeline.modifiers.append(cb)

    # Add to scene for rendering
    pipeline.add_to_scene()

    # ------------ 2) Visual styling ------------
    data = pipeline.compute()
    # Turn off cell wireframe
    data.cell.vis.enabled = False

    # Show atoms as spheres; use per-atom radius and apply global scaling
    data.particles.vis.shape = ParticlesVis.Shape.Sphere
    # Correct global radius scaling attribute:
    data.particles.vis.scaling = float(radius_scale)

    # ------------ 3) Auto camera orientation (take a,b,c by columns) ------------
    cell = np.array(data.cell)  # shape=(3,4); first 3 columns are a,b,c; last column is origin
    a = cell[:, 0]; b = cell[:, 1]; c = cell[:, 2]

    # Use an isometric-ish view direction: â + b̂ + ĉ
    dir_raw = normalize(normalize(a) + normalize(b) + normalize(c))
    camera_dir = (float(dir_raw[0]), float(dir_raw[1]), float(dir_raw[2]))

    # Up vector: start with c and orthogonalize
    up0 = c
    up_raw = normalize(up0 - np.dot(up0, dir_raw) * dir_raw)
    camera_up = (float(up_raw[0]), float(up_raw[1]), float(up_raw[2]))

    # ------------ 4) View + rendering ------------
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.camera_dir = camera_dir
    try:
        vp.camera_up = camera_up
    except Exception:
        pass

    # Auto zoom-to-fit
    vp.zoom_all()

    # Slightly push camera along -dir for a nicer margin
    L = max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))
    pos = np.array(vp.camera_pos, float)
    vp.camera_pos = tuple((pos - 0.10 * L * dir_raw).tolist())

    # Moderate field of view
    vp.fov = math.radians(45)

    # High-quality yet fast Tachyon render: white background + AO
    renderer = TachyonRenderer(
        shadows=False,
        ambient_occlusion=True,
        direct_light_intensity=1.2
    )

    ensure_dir(os.path.dirname(out_path))
    vp.render_image(
        size=(int(size[0]), int(size[1])),
        filename=out_path,
        background=(1, 1, 1),
        renderer=renderer
    )

    return out_path


class CrystalForgeAgent(Agent):
    def __init__(self, name, platform):
        super().__init__(name, platform)
        self.device = 'cuda' if TORCH_OK and torch and torch.cuda.is_available() else 'cpu'
        self.ckpt = self.platform.config.TASK2MODEL.get('crystal_forge_seg_model')
        self.model = _prepare_model_from_ckpt(self.ckpt, self.device) if TORCH_OK else None
        self.cif_lib = getattr(self.platform.config, 'CIF_LIB_DIR', 'database/cif_lib')
        self.topk = getattr(self.platform.config, 'CRYSTALFORGE_TOPK', 5)
        self.description = (
            "CrystalForge retrieval/matching agent: performs lightweight EM image segmentation (UNet++ / Otsu), "
            "filters the CIF library by element composition, scores candidates (frequency / NCC / phase correlation), "
            "renders top-K snapshots, and returns a unified dictionary output.\n\n"

            "Input formats (one of):\n"
            "- Native JSON: {\"image_path\": str, \"elements\": str|[str], \"top_k\": int?}\n"
            "- Unified dict: {text?: str, images?: str|[str], elements?: str|[str], top_k?: int}\n"
            "  • If images are provided, the first is used as image_path; elements can be 'Si,C' or ['Si','C']; top_k defaults to config.\n\n"

            "Output format (unified dict):\n"
            "{\n"
            "  ok: bool,\n"
            "  message: str,            # one-sentence summary for router/UI\n"
            "  text: str,               # Top-N list (file name / elements / score)\n"
            "  images: [str]|null,      # collage path if available; otherwise the mask overlay\n"
            "  cifs: [str]|null,        # absolute paths to top-K CIFs\n"
            "  meta: {                  # detailed metadata\n"
            "    top_cif_paths: [str], tile_image_paths: [str], gallery_path: str, elements: [str]\n"
            "  },\n"
            "  error?: {message: str, fields: [str]}   # present only on failure\n"
            "}\n\n"

            "Notes: elements can be auto-inferred from filename / history / text; segmentation weights and top_k are controlled by config "
            "(CRYSTALFORGE_TOPK / crystal_forge_seg_model); CIF library path is read from CIF_LIB_DIR."
        )

    def expected_input_schema(self) -> str:
        return '{"image_path": str, "elements": str|list?, "top_k": int?}'

    # -------------------- LLM-backed controller helpers --------------------

    def _error_dict(self, message: str, *, errors: Optional[List[str]] = None,
                    images: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {
            "ok": False,
            "text": message,
            "message": message,   # short reply to Maestro
            "error": {"message": message, "fields": errors or []},
            "images": images or None,
            "cifs": None,
            "meta": meta or None,
        }
        return out

    def controller_validate_and_fix(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and possibly auto-fix payload.
        Returns: {ok: bool, payload: dict|None, errors: [..], message: str}
        - Ensures image_path exists
        - Normalizes 'elements' (list or 'Si,C' -> list)
        - Fills missing 'elements' from filename/history/LLM when possible
        """
        errors: List[str] = []
        msg_parts: List[str] = []

        # image_path
        image_path = payload.get("image_path")
        if isinstance(image_path, str):
            try:
                image_path = os.path.abspath(image_path)
            except Exception:
                pass
        if not (isinstance(image_path, str) and os.path.isfile(image_path)):
            errors.append("image_path")
        else:
            payload["image_path"] = image_path

        # elements normalize
        elements = payload.get("elements")
        if isinstance(elements, list):
            norm = []
            for e in elements:
                if not isinstance(e, str): continue
                e = e.strip()
                if not e: continue
                e = e[0].upper() + e[1:].lower() if len(e) >= 2 else e.upper()
                if re.match(r"^[A-Z][a-z]?$", e):
                    norm.append(e)
            elements = sorted(set(norm))
        elif isinstance(elements, str):
            elements = parse_elements_text(elements)
        elif elements is None:
            # try filename
            if isinstance(image_path, str):
                base = os.path.basename(image_path)
                m = re.search(r"([A-Z][a-z]?(?:[A-Z][a-z]?)+)", base)
                if m:
                    elems = re.findall(r"[A-Z][a-z]?", m.group(1))
                    if elems:
                        elements = sorted(set(elems))
                        msg_parts.append("elements inferred from filename")
        payload["elements"] = elements

        # If still missing elements and image is valid, try history then LLM
        if (not elements) and not errors:
            inferred = self._infer_elements_from_history(payload)
            if inferred:
                payload["elements"] = inferred
                msg_parts.append("elements inferred from history")
            else:
                # Optional: LLM direct extraction from text if available in payload['text']
                txt = payload.get("text") or ""
                if isinstance(txt, str):
                    cand = re.findall(r"\b([A-Z][a-z]?)\b", txt)
                    cand = [c for c in cand if len(c) <= 2]
                    if cand:
                        payload["elements"] = sorted(set(cand))
                        msg_parts.append("elements inferred from text")

        if not payload.get("elements"):
            errors.append("elements")

        if errors:
            return {"ok": False, "payload": None, "errors": errors,
                    "message": "Missing or invalid fields: " + ", ".join(errors)}
        else:
            return {"ok": True, "payload": payload, "errors": [], "message": "; ".join(msg_parts) or "ok"}

    def controller_summarize_success(self, *, elements: List[str], top: List[dict], overlay_path: Optional[str]) -> str:
        """
        Produce a short, single-sentence message for Maestro after success.
        Uses LLM if available; otherwise returns a concise heuristic summary.
        """
        try:
            if getattr(self, 'llm', None):
                top1 = top[0] if top else {}
                prompt = (
                    "Write one short sentence (<=160 chars) summarizing crystal retrieval success for an EM image. "
                    "Mention elements and top-1 CIF filename and score. No extra commentary.\n"
                    f"Elements: {elements}\n"
                    f"Top1: name={os.path.basename(top1.get('path',''))}, score={top1.get('fused_score',0):.3f}\n"
                    f"Overlay: {bool(overlay_path)}\n"
                    "Sentence:"
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                sent = out.strip().splitlines()[0]
                return sent[:160]
        except Exception:
            pass
        # Heuristic fallback
        if top:
            return f"Retrieved Top-{len(top)} CIFs for {','.join(elements)}; best={os.path.basename(top[0]['path'])} (score={top[0].get('fused_score',0):.3f})."
        return f"Retrieved candidates for {','.join(elements)}."

    # -------------------- History-aware helpers (existing) --------------------

    def _infer_elements_from_history(self, payload: Dict[str, Any]) -> Optional[List[str]]:
        """Try to infer elements using recent agent history + (optional) LLM."""
        txt = self.recent_memory_text(10)
        m = re.findall(r"\b([A-Z][a-z]?)\b", txt)
        cand = [t for t in m if len(t) <= 2]
        if cand:
            uniq = sorted(set(cand))
            if 1 <= len(uniq) <= 6:
                return uniq
        if getattr(self, 'llm', None) is None:
            return None
        try:
            prompt = (
                "Extract valid chemical element symbols from the recent agent history.\n"
                "Return JSON: {\"elements\": [\"Si\",\"C\", ...]} (empty list if unsure).\n\n"
                f"History:\n{txt}\n\nOutput JSON:"
            )
            out = self.llm_call(messages=[{"role": "system", "content": prompt}])
            s = out.strip(); i = s.find('{'); j = s.rfind('}')
            if i != -1 and j != -1 and j > i:
                data = json.loads(s[i:j+1])
                elems = data.get('elements')
                if isinstance(elems, list) and all(isinstance(x, str) for x in elems):
                    norm = []
                    for x in elems:
                        x = x.strip()
                        if not x:
                            continue
                        x = x[0].upper() + x[1:].lower() if len(x) >= 2 else x.upper()
                        if re.match(r"^[A-Z][a-z]?$", x):
                            norm.append(x)
                    norm = sorted(set(norm))
                    if norm:
                        return norm
        except Exception:
            pass
        return None

    def _decide_continue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Kept for compatibility (now largely covered by controller_validate_and_fix)."""
        missing: List[str] = []
        image_path = payload.get('image_path')
        if not (isinstance(image_path, str) and os.path.isfile(image_path)):
            missing.append('image_path')
        elements = payload.get('elements')
        if elements is None or (isinstance(elements, str) and len(elements.strip()) == 0) or (isinstance(elements, list) and len(elements) == 0):
            missing.append('elements')
        if not missing:
            return {"continue": True, "missing": [], "message": "ok"}
        return {"continue": False, "missing": missing, "message": "Missing required fields."}

    def _infer_mask(self, image_path: str):
        lr = load_mono_as01(image_path, norm='percentile', p_lo=1.0, p_hi=99.0, auto_invert=False)
        if (self.model is None) or (not TORCH_OK):
            mask = _binarize_otsu(lr)
        else:
            probs = _sliding_window_predict(lr, self.model, tile=512, stride=None, device=self.device, amp=True)
            try:
                from skimage.filters import threshold_otsu as _thr
                thr = float(_thr(probs.numpy().flatten()))
                thr = float(max(0.05, min(0.95, thr)))
            except Exception:
                thr = 0.5
            mask = (probs >= thr).float()
        overlay = overlay_on_gray(lr, mask, alpha=0.45, color=(0, 255, 0))
        return lr, mask, overlay

    def _build_gallery(self, items: List[dict], out_dir: str) -> Tuple[str, List[str]]:
        ensure_dir(out_dir)
        tiles = []
        tile_paths = []
        for i, it in enumerate(items):
            img_path = os.path.join(out_dir, f"cif_{i+1:02d}.png")
            render_cif_snapshot(it['path'], img_path)
            tile_paths.append(img_path)

            # im = Image.open(img_path).convert('RGB')
            # draw = ImageDraw.Draw(im)
            # text = f"{os.path.basename(it['path'])} Score={it.get('fused_score',0):.3f}"
            # draw.rectangle([0, 0, im.width, 36], fill=(0,0,0,128))
            # draw.text((4, 4), text, fill=(255,255,0))
            # tiles.append(im)

            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            def _load_font(pixel_size: int) -> ImageFont.FreeTypeFont:
                """
                Prefer vector TTF fonts; fall back to default if unavailable
                (slightly less crisp).
                """
                candidates = [
                    "arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                ]
                for fp in candidates:
                    try:
                        return ImageFont.truetype(fp, pixel_size)
                    except Exception:
                        pass
                return ImageFont.load_default()

            im = Image.open(img_path).convert("RGBA")

            # If you prefer fixed tile size (example: tile_w, tile_h)
            tile_w, tile_h = 600, 600  # Adjust to your collage size
            im = im.resize((tile_w, tile_h), resample=Image.LANCZOS)  # High-quality resampling

            # Text overlay (transparent background, black text, top-centered)
            overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
            text = f"{os.path.basename(it['path'])}\nscore={it.get('fused_score',0):.3f}"

            # Prefer vector TrueType fonts to avoid bitmap jaggies
            W, H = im.width, im.height
            font_size = max(18, int(H * 0.03))
            font = _load_font(font_size)

            draw = ImageDraw.Draw(overlay)
            try:
                bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
            except AttributeError:
                bbox = draw.textbbox((0, 0), text, font=font)  # Compatibility for older Pillow

            text_w = bbox[2] - bbox[0]
            x = (im.width - text_w) // 2
            y = 15  # Top padding
            draw.multiline_text((x, y), text, font=font, fill=(0, 0, 0, 255),
                                align="center", spacing=6)

            im = Image.alpha_composite(im, overlay)

            # Optional: slight sharpening to make downscaled edges crisper (keep modest)
            im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

            tiles.append(im.convert("RGB"))

        if not tiles:
            return "", []
        cols = min(3, len(tiles)); rows = (len(tiles) + cols - 1) // cols
        w = max(t.width for t in tiles); h = max(t.height for t in tiles)
        grid = Image.new('RGB', (cols*w, rows*h), color=(20,20,20))
        for idx, im in enumerate(tiles):
            r = idx // cols; c = idx % cols
            grid.paste(im, (c*w, r*h))
        grid_path = os.path.join(out_dir, "topk_gallery.png")
        grid.save(grid_path)
        return grid_path, tile_paths

    # -------------------- Main entry --------------------

    def forward(self, messages: str | dict) -> Dict[str, Any]:
        """
        Accepts either a dict or a JSON-like string.
        Unified payload (preferred): {image_path: str, elements: list|str, top_k?: int}
        Also accepts the platform-unified {text, images, cifs, bbox, ...}, where first images[0] is taken as image_path.
        Always returns a single dictionary:
          {
            ok: bool,
            text: str,
            message: str,           # short sentence to Maestro
            images: [str]|null,     # gallery or overlay
            cifs: [str]|null,       # top-N CIF paths
            meta: { ... }|null,
            error?: {message, fields}?   # on failure
          }
        """
        # 0) parse messages -> payload
        if isinstance(messages, dict):
            payload = dict(messages)
        else:
            s = messages if isinstance(messages, str) else str(messages)
            try:
                obj = json.loads(s)
                payload = obj if isinstance(obj, dict) else {}
            except Exception:
                payload = {}
            if isinstance(s, str):
                if 'image_path' in s and 'image_path' not in payload:
                    m = re.search(r"image_path\s*[:=]\s*([\S]+)", s)
                    if m: payload['image_path'] = m.group(1)
                if 'elements' in s and 'elements' not in payload:
                    m = re.search(r"elements\s*[:=]\s*([\w,;\-\s]+)", s)
                    if m: payload['elements'] = m.group(1)

        # 1) accept unified payload and pick first image as image_path if needed
        uni = self.normalize_agent_payload(payload)
        imgs = uni.get('images')
        image_path = None
        if isinstance(imgs, list) and imgs:
            image_path = imgs[0]
        if not image_path:
            image_path = payload.get('image_path')
        if image_path:
            try:
                image_path = os.path.abspath(image_path)
            except Exception:
                pass

        # elements normalize early (string/ list / None)
        elements = uni.get('elements') if isinstance(uni, dict) else None
        if elements is None:
            elements = payload.get('elements')
        if elements is None and isinstance(uni.get('text'), str):
            cand = re.findall(r"\b([A-Z][a-z]?)\b", uni.get('text'))
            if cand:
                elements = ",".join(sorted(set([c for c in cand if len(c) <= 2])))

        top_k = int(payload.get('top_k', self.topk)) if 'top_k' in payload else self.topk

        # Build minimal payload for controller
        ctrl_payload = {
            "image_path": image_path,
            "elements": elements,
            "top_k": top_k,
            "text": uni.get("text")
        }

        # 2) LLM-backed validation & auto-fix
        ctrl = self.controller_validate_and_fix(ctrl_payload)
        if not ctrl.get("ok", False):
            # If we produced overlay earlier, attach it; here none yet, so return simple error dict
            return self._error_dict(ctrl.get("message", "Invalid input."), errors=ctrl.get("errors", []))

        # Adopt fixed payload
        fixed = ctrl["payload"]
        image_path = fixed["image_path"]
        elements = fixed["elements"]
        top_k    = int(fixed.get("top_k", self.topk))

        # 3) segmentation + overlay
        lr, mask, overlay = self._infer_mask(image_path)
        work_dir = os.path.join(self.platform.working_folder, 'artifacts')
        overlay_path = os.path.join(work_dir, 'mask_overlay.png')
        mask_path = os.path.join(work_dir, 'mask.png')
        ensure_dir(work_dir)
        overlay.save(overlay_path)
        try:
            self.remember_file(overlay_path, label='mask_overlay', kind='overlay')
        except Exception:
            pass

        mask_saved = False
        try:
            save_png01(mask_path, mask)
            mask_saved = True
            self.remember_file(mask_path, label='mask', kind='mask')
        except Exception:
            mask_saved = False
            pass

        overlay_b64 = None
        try:
            overlay_b64 = encode_image_to_b64(overlay)
        except Exception:
            overlay_b64 = None

        history_payload: Dict[str, Any] = {
            "overlay_path": overlay_path,
            "source": "crystalforge_mask_overlay",
        }
        if mask_saved:
            history_payload["mask_path"] = mask_path

        if overlay_b64:
            history_payload["images"] = [{"kind": "base64", "data": overlay_b64, "caption": "Mask overlay"}]
        else:
            history_payload["images"] = [overlay_path]

        try:
            self.platform.record_history(
                self.name,
                response="Generated mask overlay",
                history=history_payload,
            )
        except Exception:
            pass

        try:
            self.remember(
                "mask_overlay",
                payload={"image_path": image_path},
                result={"overlay_path": overlay_path, "mask_path": mask_path if mask_saved else None},
            )
        except Exception:
            pass

        # 4) retrieval
        if not os.path.isdir(self.cif_lib):
            return self._error_dict(f"Invalid CIF library path: {self.cif_lib}",
                                    images=[overlay_path] if os.path.isfile(overlay_path) else None)

        idx = build_cif_index(self.cif_lib)
        query_elems = sorted(set(elements if isinstance(elements, list) else parse_elements_text(str(elements))))
        cand = filter_by_elements(idx, query_elems)

        big = load_gray(overlay_path)
        scored = score_candidates(big, cand, num_workers=0)
        if not scored:
            return self._error_dict("No candidate CIFs found (check CIF library and element composition).",
                                    images=[overlay_path] if os.path.isfile(overlay_path) else None,
                                    meta={"elements": query_elems})

        ok_list = [r for r in scored if r.get('ok')]
        top = sorted(ok_list if ok_list else scored, key=lambda r: r.get('fused_score', -1.0), reverse=True)[:top_k]

        # 5) gallery
        gallery_dir = os.path.join(work_dir, 'crystalforge_topk')
        gallery_path, tile_paths = self._build_gallery(top, gallery_dir)
        try:
            if gallery_path:
                self.remember_file(gallery_path, label='gallery', kind='gallery')
            for tp in (tile_paths or []):
                self.remember_file(tp, label='gallery_tile', kind='gallery_tile')
        except Exception:
            pass

        # 6) build text + meta + short message for Maestro
        lines = ["CrystalForge TopN results:"]
        top_abs_paths = []
        for i, it in enumerate(top, 1):
            p_abs = os.path.abspath(it['path'])
            top_abs_paths.append(p_abs)
            elems_str = ",".join(it.get('elements', [])) if isinstance(it.get('elements'), list) else ""
            lines.append(f"{i}. {os.path.basename(it['path'])} | elems={elems_str} | score={it.get('fused_score',0):.3f}")
        # lines.append("Please select an index for the next step (default: 1)")
        text = "\n".join(lines)

        extra = {
            "top_cif_paths": top_abs_paths,
            "tile_image_paths": tile_paths,
            "gallery_path": gallery_path or "",
            "elements": query_elems,
            "mask_path": mask_path if mask_saved else "",
            "overlay_path": overlay_path,
        }

        short_msg = self.controller_summarize_success(elements=query_elems, top=top, overlay_path=overlay_path)

        out = {
            "ok": True,
            "text": text,
            "message": short_msg,  # concise reply to Maestro
            "images": [gallery_path or overlay_path] if (gallery_path or overlay_path) else None,
            "cifs": top_abs_paths or None,
            "meta": extra or None,
        }

        return out
