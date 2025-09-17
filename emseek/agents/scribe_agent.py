import os
import json
import time
from typing import Any, Dict, List, Optional

from emseek.agents.base import Agent
from emseek.protocol import to_image_refs, path_to_base64_image, is_base64_str
from emseek.utils.lang import detect_lang


class ScribeAgent(Agent):
    """
    ScribeAgent — Compose the most complete and detailed final answer for the user's
    current question using recent multi‑agent history and artifacts.

    Input (unified dict or plain text):
      { "text": str?, "images": str|[str]?, "cifs": str|[str]? }

    Output (unified dict):
      { ok: True, message: str, text: str, images: null, meta: {used_history: bool, sources?: [...], agents_seen: [...]} }
    """

    def __init__(self, name: str, platform):
        super().__init__(name, platform)
        self.description = (
            "ScribeAgent: history‑aware final composer.\n"
            "- Reads recent task history (plans + agent results).\n"
            "- Produces a thorough, well‑structured answer preserving details (images, numbers, snippets).\n"
            "- Do not propose or describe next‑step plans.\n"
            "Input: {text?: str, images?: str|[str], cifs?: str|[str]}\n"
            "Output: {ok, message, text, images: null, meta}"
        )

    def expected_input_schema(self) -> str:
        return '{"text": str?, "images": str|[str]?, "cifs": str|[str]?}'

    # -------------------- History helpers --------------------
    def _gather_history_images(self, k: int = 80) -> List[str]:
        """Collect image references from recent task history entries.

        Returns a list of image refs (not yet passed through to_image_refs). Limit to k entries scanned.
        """
        refs: List[str] = []
        task_hist = self.platform.memory.get('task_memory', []) or []
        for entry in task_hist[-k:]:
            # Skip user-submitted images; only consider agent-produced visuals
            if (entry.get('agent') or '').strip().lower() == 'user':
                continue
            resp = entry.get('response')
            if isinstance(resp, dict):
                imgs = resp.get('images') or resp.get('ref_images') or None
                if imgs is None:
                    continue
                # Normalize to base64-only refs; never expose paths
                def to_b64_ref(x) -> Optional[str]:
                    if isinstance(x, dict):
                        if x.get('kind') == 'base64' and x.get('data'):
                            return x.get('data')
                        p = x.get('path')
                        if isinstance(p, str) and os.path.isfile(p):
                            b64 = path_to_base64_image(p)
                            if b64:
                                return b64
                        return None
                    if isinstance(x, str):
                        if is_base64_str(x):
                            return x
                        if os.path.isfile(x):
                            b64 = path_to_base64_image(x)
                            if b64:
                                return b64
                        return None
                    return None

                if isinstance(imgs, list):
                    for im in imgs:
                        sref = to_b64_ref(im)
                        if sref:
                            refs.append(sref)
                else:
                    sref = to_b64_ref(imgs)
                    if sref:
                        refs.append(sref)
        return refs

    def _extract_images_from_text(self, text: str, max_images: int = 1) -> tuple[str, List[str]]:
        """Find Markdown image tags in text and return (clean_text, image_refs).

        - Converts local file paths or base64/data URLs to image refs.
        - Keeps at most max_images.
        - Replaces inline image tags in text with stable placeholders 'Image [n]: <alt>'.
        """
        import re
        if not isinstance(text, str) or not text:
            return text or "", []

        pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\)\s]+)(?:\s+['\"](?P<title>[^'\"]*)['\"])??\)")
        images: List[str] = []

        def repl(m):
            nonlocal images
            if len(images) >= max_images:
                # Drop further inline images completely from text
                return f"[Image hidden] {m.group('alt') or ''}"
            alt = (m.group('alt') or '').strip()
            src = (m.group('src') or '').strip()
            ref = None
            try:
                if not src:
                    ref = None
                elif is_base64_str(src):
                    ref = src
                else:
                    # treat as path if exists
                    try_path = src
                    if os.path.isfile(try_path):
                        b64 = path_to_base64_image(try_path)
                        if b64:
                            ref = b64
                        else:
                            ref = None
                    else:
                        # Not a local file; do not include remote URLs to avoid leaking addresses
                        ref = None
            except Exception:
                ref = None

            if ref is not None:
                images.append(ref)
                idx = len(images) - 1
                return f"Image [{idx}]: {alt}" if alt else f"Image [{idx}]"
            # If cannot resolve, keep alt text
            return alt or "[image]"

        new_text = pattern.sub(repl, text)
        return new_text, images

    def _normalize_out_images(self, images: List[str], max_images: int = 3) -> List[str]:
        """Ensure outgoing images are base64 strings only and limited to max_images."""
        out: List[str] = []
        for im in images[:max_images]:
            try:
                if isinstance(im, str):
                    if is_base64_str(im):
                        out.append(im)
                    elif os.path.isfile(im):
                        b64 = path_to_base64_image(im)
                        if b64:
                            out.append(b64)
            except Exception:
                continue
        return out[:max_images]

    def _mask_file_paths_in_text(self, text: str) -> str:
        """Best-effort masking of file paths/URLs from text output."""
        import re
        if not isinstance(text, str) or not text:
            return text or ""
        # file:// URIs
        text = re.sub(r"file://[^\s)]+", "[local-file]", text)
        # Windows paths like C:\\...\\file.png
        text = re.sub(r"[A-Za-z]:\\\\[^\s\n\r]+", "[local-file]", text)
        # Unix-like absolute paths /.../file.png
        text = re.sub(r"/(?:[^\s\n\r/]+/)+[^\s\n\r]+\.(?:png|jpg|jpeg|gif|bmp|tif|tiff|cif|pdf|txt)", "[local-file]", text, flags=re.IGNORECASE)
        # Relative paths ./foo/bar.png or ../
        text = re.sub(r"(?:\./|\.\./)[^\s\n\r]+\.(?:png|jpg|jpeg|gif|bmp|tif|tiff|cif|pdf|txt)", "[local-file]", text, flags=re.IGNORECASE)
        return text
    def _compact_history_lines(self, k: int = 80) -> List[str]:
        """Return detailed recent task history blocks from the platform without aggressive truncation.

        Includes message/text content and indicates images when present. Keeps numbers and newlines.
        """
        lines: List[str] = []
        task_hist = self.platform.memory.get('task_memory', []) or []
        for entry in task_hist[-k:]:
            ts = entry.get('timestamp', '?')
            ag = entry.get('agent', '?')
            resp = entry.get('response')

            # Build a rich block per entry
            block_parts: List[str] = [f"{ts} - {ag}:"]
            if isinstance(resp, dict):
                msg = resp.get('message')
                txt = resp.get('text')
                if isinstance(msg, str) and msg.strip():
                    block_parts.append("Message:")
                    block_parts.append(msg.strip())
                if isinstance(txt, str) and txt.strip():
                    block_parts.append("Text:")
                    block_parts.append(txt.strip())

                # Indicate images in response if any
                imgs = resp.get('images') or resp.get('ref_images') or None
                if imgs is not None:
                    try:
                        # Avoid dumping raw base64; show counts and captions/paths
                        if isinstance(imgs, list):
                            img_summ: List[str] = []
                            for i, im in enumerate(imgs):
                                if isinstance(im, dict):
                                    kind = im.get('kind') or ('base64' if im.get('data') else 'path')
                                    cap = im.get('caption') or ''
                                    if kind == 'base64':
                                        ln = len(im.get('data', ''))
                                        img_summ.append(f"  - [{i}] {kind} (len={ln}) {('— ' + cap) if cap else ''}")
                                    else:
                                        # Do not reveal concrete file paths
                                        img_summ.append(f"  - [{i}] {kind} [local-file] {('— ' + cap) if cap else ''}")
                                elif isinstance(im, str):
                                    # path or base64 heuristic
                                    img_summ.append(f"  - [{i}] string ({'base64' if len(im) > 64 else 'local-file'})")
                        if img_summ:
                            block_parts.append("Images:")
                            block_parts.extend(img_summ)
                        else:
                            # Single image ref
                            block_parts.append(f"Images: 1 item ({type(imgs).__name__})")
                    except Exception:
                        pass

                # Preserve key numeric/meta fields when small
                meta = resp.get('meta')
                if isinstance(meta, dict) and meta:
                    try:
                        # Keep short meta (<= 1200 chars) inline for detail preservation
                        meta_str = json.dumps(meta, ensure_ascii=False)
                        if len(meta_str) <= 1200:
                            block_parts.append("Meta:")
                            block_parts.append(meta_str)
                    except Exception:
                        pass
            else:
                content = str(resp) if resp is not None else ''
                if content:
                    block_parts.append(content)

            lines.append("\n".join([p for p in block_parts if isinstance(p, str) and p != '']))
        return lines

    def _extract_sources(self) -> List[Dict[str, Any]]:
        """Best‑effort collection of sources/citations from agent metas."""
        sources: List[Dict[str, Any]] = []
        task_hist = self.platform.memory.get('task_memory', []) or []
        for entry in task_hist[-80:]:
            resp = entry.get('response')
            if isinstance(resp, dict):
                meta = resp.get('meta') or {}
                # Common meta keys that may contain sources
                for key in ('sources', 'refs', 'citations', 'provenance'):
                    val = meta.get(key)
                    if isinstance(val, list) and val:
                        try:
                            sources.extend(val[:10])
                        except Exception:
                            pass
        # de‑duplicate simple dicts/strings
        seen = set()
        unique: List[Dict[str, Any]] = []
        for s in sources:
            try:
                key = json.dumps(s, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = str(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    # -------------------- Main entry --------------------
    def forward(self, messages: str | dict) -> Dict[str, Any]:
        # Tolerant parse
        if isinstance(messages, dict):
            raw = dict(messages)
        else:
            s = messages if isinstance(messages, str) else str(messages)
            try:
                obj = json.loads(s)
                raw = obj if isinstance(obj, dict) else {"text": s}
            except Exception:
                raw = {"text": s}

        # Unified payload for optional future expansion (keeps it flat)
        uni = self.normalize_agent_payload(raw)
        question = (uni.get('text') or raw.get('text') or raw.get('query') or '').strip()
        opts = uni.get('options') or {}
        # Determine how many images to include in final answer (bounded 1..6)
        try:
            cfg_topk = int(getattr(self.platform.config, 'FINAL_IMAGES_TOPK', 3))
        except Exception:
            cfg_topk = 3
        try:
            req_topk = int(opts.get('top_k')) if isinstance(opts.get('top_k'), (int, str)) else None
        except Exception:
            req_topk = None
        max_images = req_topk if isinstance(req_topk, int) and req_topk > 0 else cfg_topk
        if max_images < 1:
            max_images = 1
        if max_images > 6:
            max_images = 6

        # Gather compact recent history
        lines = self._compact_history_lines(80)
        history_text = "\n".join(lines)
        sources = self._extract_sources()
        agents_seen = list({(e.get('agent') or '?') for e in (self.platform.memory.get('task_memory') or [])})

        # Build prompt
        lang = detect_lang(question)
        if not question:
            question = "Please provide a comprehensive summary based on the history"

        # English-only system instruction
        sys_inst = (
            "You are ScribeAgent, responsible for composing the final answer.\n"
            "Use the provided history to preserve as many details as possible, including exact numbers, identifiers, and quoted snippets.\n"
            "When images are relevant, refer to them generically as 'Image [n]' with optional captions. Do NOT include file paths, URLs, or storage locations.\n"
            "Do NOT suggest or describe next-step plans. Provide only the final answer with supporting evidence from the history.\n"
            "Present the answer in a clear, well-structured, professional tone."
        )

        # LLM call (fallbacks if LLM not available)
        composed = None
        if getattr(self, 'llm', None):
            messages_llm = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": ("User question: " + question)},
                {"role": "user", "content": ("History:\n" + history_text)},
            ]
            try:
                composed = self.llm_call(messages=messages_llm)
            except Exception:
                composed = None

        if not composed:
            # Fallback: include question and detailed history blocks without truncation
            header = "Final Answer"
            composed = f"{header}:\n{question}\n\nDetails from recent history:\n{history_text}"

        # Extract inline images from composed text and gather minimal necessary images
        text_clean, images_from_text = self._extract_images_from_text(composed or "", max_images=max_images)
        text_clean = self._mask_file_paths_in_text(text_clean)
        history_image_refs = self._gather_history_images(80)
        # Convert and limit images (prefer those explicitly referenced in text)
        out_images: List[str] = []
        if images_from_text:
            out_images.extend(images_from_text[:max_images])
        if len(out_images) < max_images and history_image_refs:
            # Fill remaining slots with latest unique refs from history (agent-produced only)
            for ref in reversed(history_image_refs):
                if len(out_images) >= max_images:
                    break
                # Deduplicate by value
                key = str(ref)
                if any((str(r) == key) for r in out_images):
                    continue
                out_images.append(ref)
        out_images = self._normalize_out_images(out_images, max_images=max_images)

        meta = {
            "used_history": True,
            "agents_seen": agents_seen,
            "sources": sources if sources else None,
        }

        try:
            self.remember("forward", payload={"text": question}, result={"len": len(composed or '')})
        except Exception:
            pass

        return {
            "ok": True,
            "message": "Final answer composed from detailed history",
            "text": text_clean or "",
            "images": out_images or None,
            "meta": meta,
        }
