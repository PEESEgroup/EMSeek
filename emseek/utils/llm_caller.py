import os
from typing import List, Any

import litellm


class LLMCaller:
    def __init__(self, config):
        """
        LLM caller that routes all chat/embedding calls through LiteLLM.
        Uses `config.LLM_MODEL` for chat and `config.EMBEDDING_MODEL` for embeddings
        (falls back to a reasonable default if not set).
        """
        self.config = config

    def forward(self, messages=None, max_tokens=None, temperature=None, **kwargs) -> str:
        if messages is None:
            raise ValueError("The 'messages' parameter is required.")
        model = getattr(self.config, 'LLM_MODEL', None) or os.getenv('LLM_MODEL') or 'gpt-5-nano-2025-08-07'
        # Resolve centralized defaults from config/env if not provided
        if max_tokens is None:
            max_tokens = getattr(self.config, 'LLM_MAX_TOKENS', None)
            if max_tokens is None:
                env_v = os.getenv('LLM_MAX_TOKENS', None)
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
            tcfg = getattr(self.config, 'LLM_TEMPERATURE', None)
            if tcfg is not None:
                temperature = tcfg
            else:
                env_t = os.getenv('LLM_TEMPERATURE', None)
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
        call_args = dict(model=model, messages=messages)
        if max_tokens is not None:
            call_args['max_tokens'] = max_tokens
        if temperature is not None:
            call_args['temperature'] = temperature
        resp = litellm.completion(**{**call_args, **kwargs})
        # LiteLLM returns an OpenAI-style response (dict-like or object). Extract content robustly.
        try:
            # object-like
            return resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            try:
                # dict-like
                return resp['choices'][0]['message']['content']  # type: ignore[index]
            except Exception:
                return str(resp)

    def get_embedding(self, text: str, model: str = None) -> List[float]:
        text = (text or '').replace('\n', ' ')
        emb_model = model or getattr(self.config, 'EMBEDDING_MODEL', None) or os.getenv('EMBEDDING_MODEL') or 'text-embedding-3-small'
        try:
            resp = litellm.embedding(model=emb_model, input=[text])
            try:
                return resp.data[0].embedding  # type: ignore[attr-defined]
            except Exception:
                return resp['data'][0]['embedding']  # type: ignore[index]
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
