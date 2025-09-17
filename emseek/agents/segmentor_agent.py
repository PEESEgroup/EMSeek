import io
import os
import json
import base64
from typing import Dict, Optional, Tuple, Any, List

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

from emseek.utils.llm_caller import LLMCaller
from emseek.utils.mllm_caller import MLLMCaller
from emseek.agents import Agent
from emseek.models import UnetPlusPlus
from emseek.utils import (
    overlay_mask_on_image,
    convert_yolo_to_bbox,
    crop_patch,
    save_image_file,
)
from emseek.utils.input import is_dataurl_or_b64


class SegMentorAgent(Agent):
    def __init__(self, name: str, platform, caller_type: str = "mllm"):
        """
        SegMentorAgent — EM image segmentation + expert description with an LLM-backed controller.

        - Validates/repairs inputs (image path/base64, optional bbox) via controller.
        - Runs Unet++ (patch-guided) to predict mask and overlay.
        - Uses M-LLM to produce a concise, domain-appropriate description (with an optional refined query).
        - Returns a unified dict compatible with Maestro.

        Success output:
          { ok: True, message, text, images, meta }
        Failure output:
          { ok: False, message, error{message, fields}, images? }
        """
        super().__init__(name, platform)
        self.caller_type = caller_type
        self.mllm_caller = MLLMCaller(self.platform.config)
        self.llm_caller = LLMCaller(self.platform.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Unet++ with prompt channel (patch-guided)
        self.model = UnetPlusPlus(encoder_name="resnet18", prompt=True, classes=2).to(self.device)
        self.model.eval()
        # Track which checkpoint is currently loaded to avoid redundant loads
        self._loaded_checkpoint_path: Optional[str] = None
        self._loaded_model_key: Optional[str] = None

        self.em_images: Dict[str, str] = {}

        self.prompt = (
            "You are a domain expert in electron microscopy image analysis.\n"
            "Given the provided EM image (with mask overlay), produce a precise, professional description that covers:\n"
            "- Global morphology and salient structures (grain boundaries, particles, pores, lattice contrast);\n"
            "- Contrast/texture clues related to imaging conditions (qualitative, no speculation beyond image);\n"
            "- Mask/overlay interpretation: what regions are likely highlighted and why;\n"
            "- Any visible spatial patterns, anomalies, or artifacts that may impact analysis.\n"
            "Keep it concise, objective, and technically sound."
        )
        # --- Description---
        self.description = (
            "SegMentorAgent: binary EM image segmentation with a vision-LLM technical summary.\n"
            "Input (unified dict): {text?: str, images?: str|[str], image_path?: str, bbox?: {x,y,w,h}?, model?: str?}\n"
            "- images/image_path: local file path or base64/dataURL; if both are missing, attempts to infer a recent local path; only the first image is used.\n"
            "- bbox: YOLO-normalized coordinates (x,y,w,h in [0,1]), optional.\n"
            "Scope:\n"
            "- Validate and lightly repair inputs (ignore malformed bbox).\n"
            "- Run Unet++ for foreground/background segmentation; when bbox is provided, use patch-guided prompting; produce a binary mask and an overlay.\n"
            "- Load weights via 'model': a key in cfg.TASK2MODEL or a direct checkpoint path; if unresolved, keep current weights.\n"
            "- Use a multimodal LLM to generate a concise, objective description of the overlay; if text is provided, refine it into a short prompt first.\n"
            "Output:\n"
            "- Success: {ok: True, message, text, images: [overlay_path], meta:{refined_query, original_image, mask_path, overlay_path, bbox_image?, patch_path?, model?, checkpoint?}}\n"
            "- Failure: {ok: False, message, error:{message, fields}, images?}.\n"
            "Limits (out of scope): no training/super-resolution/inpainting/quantitative measurement; no material/phase identification or assertive conclusions; no batch multi-image (first only); no 3D/sequence segmentation; outputs only a pixel-level binary mask and overlay (no morphology post-processing or vector annotations)."
        )

    def expected_input_schema(self) -> str:
        return '{"text": str?, "images": str|[str]?, "image_path": str?, "bbox": {"x":float,"y":float,"w":float,"h":float}?, "model": str?}'

    # -------------------------- Controller helpers --------------------------

    def _error_dict(self, message: str, *, errors: Optional[List[str]] = None,
                    images: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {
            "ok": False,
            "text": message,
            "message": message,
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
        Validate/repair inputs for segmentation:
          - Accept image as:
              * unified images[0] (path or base64/dataURL)
              * image_path (absolute/relative)
          - Optional bbox (YOLO normalized dict) => we keep original and convert later.
          - Optionally infer from recent memory via LLM if nothing provided.
        Returns: {ok, payload|None, errors, message}
        """
        errors: List[str] = []
        msg_parts: List[str] = []

        # 1) Resolve image source
        img = payload.get("images")
        image_path = payload.get("image_path")
        image_str: Optional[str] = None

        # Normalize images field
        if isinstance(img, list) and img:
            if isinstance(img[0], str):
                image_str = img[0]
        elif isinstance(img, str):
            image_str = img

        # Prefer image_str; if it looks like a path, keep path; if it looks like base64/dataURL, keep base64
        image_kind = None  # "path" | "base64"

        # If unified image missing, fallback to image_path
        if image_str is None and isinstance(image_path, str):
            image_str = image_path

        # 2) If still none, try memory
        if image_str is None:
            # quick scan of remembered files
            candidates = (
                self.em_images.get("EM Image with Mask"),
                self.em_images.get("EM Image"),
            )
            for p in candidates:
                if p and os.path.isfile(p):
                    image_str = p; msg_parts.append("image inferred from memory"); break

        # 3) If still none, ask LLM from history
        if image_str is None and getattr(self, 'llm', None):
            try:
                hist = self.recent_memory_text(8)
                prompt = (
                    "You validate inputs for an image segmentation agent.\n"
                    "From the recent history, extract a usable local image path if available.\n"
                    "Return JSON {\"image_path\": \"/abs/path\"} or {\"image_path\": null}.\n\n"
                    f"History:\n{hist}\n\nOutput JSON:"
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                s = out.strip(); i = s.find('{'); j = s.rfind('}')
                if i != -1 and j != -1 and j > i:
                    data = json.loads(s[i:j+1])
                    cp = data.get('image_path')
                    if isinstance(cp, str) and os.path.isfile(cp):
                        image_str = os.path.abspath(cp)
                        msg_parts.append("image inferred from history")
            except Exception:
                pass

        # 4) Validate the chosen image
        if isinstance(image_str, str):
            if is_dataurl_or_b64(image_str):
                image_kind = "base64"
            else:
                # treat as path
                try:
                    image_str = os.path.abspath(image_str)
                except Exception:
                    pass
                if os.path.isfile(image_str):
                    image_kind = "path"
                else:
                    image_kind = None

        if image_kind is None:
            errors.append("image")
            return {"ok": False, "payload": None, "errors": errors,
                    "message": "Missing or invalid image (path or base64/dataURL)."}

        # 5) BBox is optional; keep as-is (YOLO normalized)
        bbox = payload.get("bbox")
        if bbox is not None and not isinstance(bbox, dict):
            # Ignore malformed bbox instead of failing the whole request
            bbox = None
            msg_parts.append("bbox ignored (malformed)")

        fixed = {
            "image_kind": image_kind,
            "image": image_str,
            "bbox": bbox,
            "text": payload.get("text") or "",
        }
        return {"ok": True, "payload": fixed, "errors": [], "message": "; ".join(msg_parts) or "ok"}

    def controller_summarize_success(self, *, had_bbox: bool, overlay_path: Optional[str], desc_len: int) -> str:
        """
        Produce a short one-liner for Maestro after success (<=160 chars).
        """
        try:
            if getattr(self, 'llm', None):
                prompt = (
                    "Write a single concise sentence (<=160 chars) summarizing a successful EM image segmentation and description.\n"
                    f"bbox_used={had_bbox}, overlay={bool(overlay_path)}, desc_len={desc_len}."
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                return out.strip().splitlines()[0][:160]
        except Exception:
            pass
        # Fallback
        base = "Segmentation completed; overlay generated"
        if had_bbox:
            base += " with a reference patch"
        base += f"; description length={desc_len}."
        return base[:160]

    # -------------------------- Core inference helpers --------------------------

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights (robust to common checkpoint formats) and switch to eval mode.

        Supports:
          - raw state_dict
          - dicts with 'state_dict' or 'model_state_dict'
          - DataParallel 'module.'-prefixed keys (stripped)
        """
        if not checkpoint_path:
            return
        # Skip if already loaded
        if self._loaded_checkpoint_path and os.path.abspath(checkpoint_path) == os.path.abspath(self._loaded_checkpoint_path):
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Extract possible nested state dicts
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Might already be a state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint  # fallback

            # Strip 'module.' prefix if present
            def _strip_module(sd):
                needs_strip = any(k.startswith('module.') for k in sd.keys())
                if not needs_strip:
                    return sd
                return {k[len('module.'):]: v for k, v in sd.items()}

            state_dict = _strip_module(state_dict)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from: {checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}")
            self._loaded_checkpoint_path = os.path.abspath(checkpoint_path)
        except Exception as e:
            print(f"[SegMentor] Failed to load checkpoint '{checkpoint_path}': {e}")
        finally:
            self.model.eval()

    def _resolve_model_checkpoint(self, model_key: Optional[str]) -> Tuple[Optional[str], str]:
        """Resolve model key or path to a checkpoint path via cfg.TASK2MODEL.

        Returns: (checkpoint_path_or_none, resolved_key)
          - If model_key is a known key in TASK2MODEL, use it.
          - If model_key is an existing file path, use it as-is with key='custom'.
          - Otherwise, fall back to 'general_model'.
        """
        cfg = self.platform.config
        task2model = getattr(cfg, 'TASK2MODEL', {}) or {}
        resolved_key = None
        ckpt_path: Optional[str] = None

        # Preferred: explicit key in mapping
        if isinstance(model_key, str) and model_key in task2model:
            resolved_key = model_key
            ckpt_path = task2model.get(model_key)
        # If user passed a direct path
        elif isinstance(model_key, str) and os.path.isfile(model_key):
            resolved_key = 'custom'
            ckpt_path = model_key
        else:
            # Default fallback
            if 'general_model' in task2model:
                resolved_key = 'general_model'
                ckpt_path = task2model.get('general_model')
            else:
                resolved_key = model_key or 'unknown'
                ckpt_path = None

        if ckpt_path:
            try:
                ckpt_path = os.path.abspath(ckpt_path)
            except Exception:
                pass

        return ckpt_path, resolved_key or ''

    def preprocess_image(self, image: Image.Image, input_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
        """Preprocess image: ensure RGB, resize, normalize, and add batch dim."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        image = image.convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return preprocess(image).unsqueeze(0)
    
    def _make_default_patch(self, image: Image.Image, size: Tuple[int, int] = (64, 64)) -> Image.Image:
        """Create a default patch (center crop) when no bbox is provided."""
        w, h = image.size
        cw, ch = size
        left = max(0, (w - cw) // 2)
        top = max(0, (h - ch) // 2)
        return image.crop((left, top, min(w, left + cw), min(h, top + ch)))

    def predict(self, image: Image.Image, patch: Optional[Image.Image]) -> np.ndarray:
        """Run inference on the image with an optional reference patch and return a mask."""
        image_tensor = self.preprocess_image(image, input_size=(1024, 1024)).to(self.device)
        if patch is None:
            # patch = self._make_default_patch(image)
            patch_tensor = None
        else:
            patch_tensor = self.preprocess_image(patch, input_size=(64, 64)).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor, patch_tensor)
            pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        return pred_mask

    # -------------------------- Main entry (new forward) --------------------------

    def forward(self, args: Dict[str, Any] | str) -> Dict[str, Any]:
        """
        LLM-controlled end-to-end segmentation & description.

        Accepts (unified):
          {text?: str, images?: str|[str], image_path?: str, bbox?: {x,y,w,h}?}

        Returns:
          success: {ok: True, message, text, images:[overlay], meta:{...}}
          failure: {ok: False, message, error{message, fields}, images?}
        """
        # 0) parse args (tolerant)
        if isinstance(args, dict):
            raw = dict(args)
        else:
            s = args if isinstance(args, str) else str(args)
            try:
                obj = json.loads(s)
                raw = obj if isinstance(obj, dict) else {"text": s}
            except Exception:
                raw = {"text": s}

        # 1) normalize top-level via platform helper (keeps it flat)
        uni = self.normalize_agent_payload(raw)
        payload = {
            "text": uni.get("text") if isinstance(uni.get("text"), str) else raw.get("text"),
            "images": uni.get("images") if uni.get("images") is not None else raw.get("images"),
            "image_path": uni.get("image_path") or raw.get("image_path"),
            "bbox": uni.get("bbox") or raw.get("bbox"),
            # Carry through 'model' selection (may be a known key or a direct path)
            "model": uni.get("model") or raw.get("model"),
        }

        # 2) controller validate & repair
        ctrl = self.controller_validate_and_fix(payload)
        if not ctrl.get("ok", False):
            return self._error_dict(ctrl.get("message", "Invalid input."), errors=ctrl.get("errors", ["image"]))

        fixed = ctrl["payload"]
        image_kind: str = fixed["image_kind"]
        image_src: str = fixed["image"]
        bbox = fixed.get("bbox")
        query_text = fixed.get("text") or ""

        # 2.5) Select and load model according to provided 'model' key/path
        model_key = payload.get("model")
        ckpt_path, resolved_key = self._resolve_model_checkpoint(model_key)
        if ckpt_path:
            # Load only if changed
            if os.path.abspath(ckpt_path) != (self._loaded_checkpoint_path or ""):
                self.load_model(ckpt_path)
                self._loaded_model_key = resolved_key
                try:
                    self.remember("load_model", payload={"model": resolved_key, "path": ckpt_path}, result="ok")
                except Exception:
                    pass
        else:
            # No checkpoint resolved; proceed with existing weights
            try:
                self.remember("load_model", payload={"model": model_key, "path": None}, result="fallback")
            except Exception:
                pass

        # 3) decode/load image -> PIL
        try:
            if image_kind == "base64":
                # dataURL or raw base64
                data = image_src.split(",", 1)[1] if image_src.strip().startswith("data:image/") else image_src
                image_bytes = base64.b64decode(data)
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                image_pil = Image.open(image_src).convert("RGB")
        except Exception:
            return self._error_dict("Failed to open the provided image.", errors=["image"])

        original_image = image_pil.copy()

        # 4) derive patch from bbox if provided
        patch = None
        patch_path = None
        bbox_img_path = None
        if isinstance(bbox, dict):
            try:
                bbox_coords = convert_yolo_to_bbox(bbox, original_image.size)
                patch = crop_patch(original_image, bbox_coords)
                patch_path = save_image_file(patch, self.platform.working_folder, "patch")
                self.em_images["Target Image"] = patch_path
                try:
                    self.remember_file(patch_path, label="Target Image")
                except Exception:
                    pass

                draw = ImageDraw.Draw(image_pil)
                draw.rectangle([bbox_coords[0], bbox_coords[1]], outline="red", width=2)
                bbox_img_path = save_image_file(image_pil, self.platform.working_folder, "bbox")
                self.em_images["EM Images with Target bbox"] = bbox_img_path
                try:
                    self.remember_file(bbox_img_path, label="EM Images with Target bbox")
                except Exception:
                    pass
            except Exception:
                # ignore bbox failure; continue without patch
                patch = None

        # 5) persist original
        original_path = save_image_file(original_image, self.platform.working_folder, "original")
        self.em_images["EM Image"] = original_path
        try:
            self.remember_file(original_path, label="EM Image")
        except Exception:
            pass

        # 6) segmentation
        try:
            mask = self.predict(original_image, patch)
        except Exception:
            return self._error_dict("Segmentation failed (model inference error).", errors=["internal"])

        mask_path = save_image_file(mask, self.platform.working_folder, "mask")
        self.em_images["Mask"] = mask_path
        try:
            self.remember_file(mask_path, label="Mask")
        except Exception:
            pass

        mask_overlay = overlay_mask_on_image(original_image, mask)
        overlay_path = save_image_file(mask_overlay, self.platform.working_folder, "mask_overlay")
        self.em_images["EM Image with Mask"] = overlay_path
        try:
            self.remember_file(overlay_path, label="EM Image with Mask")
        except Exception:
            pass

        # 7) refine query (short)
        refined_query = ""
        try:
            rq_messages = [
                {
                    "role": "system",
                    "content": (
                        "Refine the user's query to match the target EM image; return a short, direct prompt "
                        "for image-grounded description. If the user text is empty, return 'Describe the image.'"
                    ),
                },
                {"role": "user", "content": f"User text: {query_text or ''}"},
            ]
            refined_query = self.llm_caller.forward(messages=rq_messages)
            if not isinstance(refined_query, str) or not refined_query.strip():
                refined_query = "Describe the image."
        except Exception:
            refined_query = "Describe the image."

        # 8) produce multimodal description
        try:
            mm_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a recognized expert in electron microscopy image analysis and in producing "
                        "detailed, objective technical descriptions."
                    ),
                },
                {"role": "user", "content": self.prompt},
                {"role": "user", "content": refined_query},
            ]
            description = self.mllm_caller.forward(messages=mm_messages, image_path=overlay_path)
        except Exception:
            description = "Segmentation completed; the system could not generate an automatic description."

        # 9) success one-liner for Maestro
        short_msg = self.controller_summarize_success(
            had_bbox=isinstance(bbox, dict),
            overlay_path=overlay_path,
            desc_len=len(description or ""),
        )

        # 10) assemble meta and output
        meta = {
            "refined_query": refined_query,
            "original_image": original_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "bbox_image": bbox_img_path,
            "patch_path": patch_path,
            "model": self._loaded_model_key,
            "checkpoint": self._loaded_checkpoint_path,
        }

        out = {
            "ok": True,
            "message": short_msg,
            "text": description,
            "images": [overlay_path] if overlay_path else [original_path],
            "meta": meta,
        }

        try:
            self.remember("forward", payload={"text": query_text}, result={"message": short_msg, "overlay": overlay_path})
        except Exception:
            pass

        return out
