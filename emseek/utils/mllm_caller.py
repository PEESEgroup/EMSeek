import os
import base64
from typing import Any, Optional

import litellm

try:
    import requests  # Lightweight fetch for image URLs
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime_from_name(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".gif"):
        return "image/gif"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".bmp"):
        return "image/bmp"
    # default
    return "image/jpeg"


def _fetch_url_b64(url: str) -> Optional[str]:
    """Fetch image at URL and return raw base64 string. Returns None on failure."""
    if not url:
        return None
    # If already a data URL, return without prefix removal
    if url.startswith("data:"):
        try:
            return url.split(",", 1)[1]
        except Exception:
            return None
    # Try requests first, fall back to urllib
    try:
        if requests is not None:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
    except Exception:
        pass
    try:
        import urllib.request as _u
        with _u.urlopen(url, timeout=10) as r:  # type: ignore
            data = r.read()
            return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


class MLLMCaller:
    def __init__(self, config):
        """
        Multimodal caller via LiteLLM. Uses `config.MLLM_MODEL` for vision-capable models.
        """
        self.config = config

    def forward(self, messages=None, image_path=None, image_url=None, image_b64=None,
                max_tokens=None, temperature=None, **kwargs) -> str:
        if messages is None:
            raise ValueError("The 'messages' parameter is required.")

        # Prepare content from the last message only if we need to combine with an image
        content: Any = messages[-1]['content'] if messages else ''
        if isinstance(content, list):
            base_content_list = content
        elif isinstance(content, str):
            base_content_list = [{"type": "text", "text": content}]
        else:
            base_content_list = [{"type": "text", "text": str(content)}]

        img_data_url = None
        # Normalize provided image into a base64 data URL
        if image_b64:
            # Accept either raw base64 or data URL
            if isinstance(image_b64, str) and image_b64.startswith("data:"):
                img_data_url = image_b64
            else:
                img_data_url = f"data:image/jpeg;base64,{image_b64}"
        elif image_path:
            mime = _guess_mime_from_name(str(image_path))
            img_data_url = f"data:{mime};base64,{encode_image(image_path)}"
        elif image_url:
            # Convert remote URL to base64 for consistent handling
            b64 = _fetch_url_b64(image_url)
            if b64:
                mime = _guess_mime_from_name(str(image_url))
                img_data_url = f"data:{mime};base64,{b64}"
            else:
                # As a last resort, pass through if the URL is a data URL
                if isinstance(image_url, str) and image_url.startswith("data:"):
                    img_data_url = image_url

        # Build final messages ensuring image parts live under a 'user' role
        final_messages = list(messages)
        if img_data_url:
            last_role = messages[-1]['role'] if messages else 'user'
            image_part = {"type": "image_url", "image_url": {"url": img_data_url}}
            if last_role == 'user':
                # Merge image into the last user message
                merged = list(base_content_list) if isinstance(base_content_list, list) else [{"type": "text", "text": str(base_content_list)}]
                merged.append(image_part)
                final_messages = list(messages[:-1]) + [{"role": "user", "content": merged}]
            else:
                # Keep system/assistant messages untouched; append a new user message with the image
                final_messages = list(messages) + [{"role": "user", "content": [image_part]}]
        else:
            # No image provided; keep original messages unchanged
            final_messages = list(messages)

        model = getattr(self.config, 'MLLM_MODEL', None) or os.getenv('MLLM_MODEL') or 'gpt-4o-mini'

        # Resolve centralized defaults from config/env if not provided
        if max_tokens is None:
            max_tokens = getattr(self.config, 'MLLM_MAX_TOKENS', None)
            if max_tokens is None:
                env_v = os.getenv('MLLM_MAX_TOKENS', None)
                if env_v is not None:
                    s = env_v.strip().lower()
                    if s == "" or s in {"none", "null", "default", "auto"}:
                        max_tokens = None
                    else:
                        try:
                            max_tokens = int(env_v)
                        except Exception:
                            max_tokens = None
        if temperature is None:
            tcfg = getattr(self.config, 'MLLM_TEMPERATURE', None)
            if tcfg is not None:
                temperature = tcfg
            else:
                env_t = os.getenv('MLLM_TEMPERATURE', None)
                if env_t is not None:
                    s = env_t.strip().lower()
                    if s == "" or s in {"none", "null", "default", "auto"}:
                        temperature = None
                    else:
                        try:
                            temperature = float(env_t)
                        except Exception:
                            temperature = None

        # Only pass params that are not None (to use LiteLLM defaults otherwise)
        call_args = dict(model=model, messages=final_messages)
        if max_tokens is not None:
            call_args['max_tokens'] = max_tokens
        if temperature is not None:
            call_args['temperature'] = temperature
        resp = litellm.completion(**{**call_args, **kwargs})
        try:
            return resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            try:
                return resp['choices'][0]['message']['content']  # type: ignore[index]
            except Exception:
                return str(resp)
