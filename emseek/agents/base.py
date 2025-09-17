import os
import json
import time
from typing import Any, Dict, Tuple, Optional
from contextlib import contextmanager

try:
    from emseek.utils.llm_caller import LLMCaller
except Exception:
    LLMCaller = None  # type: ignore
try:
    from emseek.utils.mllm_caller import MLLMCaller
except Exception:
    MLLMCaller = None  # type: ignore

class Agent:
    def __init__(self, name, platform):
        self.name = name
        self.platform = platform
        # Per-agent internal memory for local reasoning/history
        # Each entry: {timestamp, event, payload?, result?}
        self.memory = []  # type: ignore[var-annotated]
        # Lightweight guard LLM for input validation/normalization (optional)
        self._guard = LLMCaller(self.platform.config) if LLMCaller else None
        # General-purpose LLM handle available to all agents (optional)
        self.llm = LLMCaller(self.platform.config) if LLMCaller else None
        # General-purpose multimodal LLM handle (optional)
        self.mllm = MLLMCaller(self.platform.config) if 'MLLMCaller' in globals() and MLLMCaller else None

    # -------------------- Internal Memory --------------------
    def remember(self, event: str, payload: Any = None, result: Any = None) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "payload": payload,
            "result": result,
        }
        self.memory.append(entry)
        # Append agent-scoped JSONL log for easier inspection
        try:
            logs_dir = os.path.join(self.platform.working_folder, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            safe_name = str(self.name).replace("/", "_").replace(" ", "_")
            log_path = os.path.join(logs_dir, f"agent_{safe_name}.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def remember_file(self, path: str, label: Optional[str] = None, kind: str = "artifact") -> None:
        """Record a generated/saved file path into history/memory."""
        if not path:
            return
        try:
            abspath = os.path.abspath(path)
        except Exception:
            abspath = path
        # intentionally no-op for history logging

    def recent_memory_text(self, k: int = 10) -> str:
        lines = []
        for e in self.memory[-k:]:
            evt = e.get("event", "?")
            payload = e.get("payload")
            lines.append(f"{e.get('timestamp','?')} - {evt}: {str(payload)[:200]}")
        return "\n".join(lines)

    # -------------------- IO Normalization --------------------
    def expected_input_schema(self) -> str:
        """
        Override in subclasses to describe expected JSON keys succinctly.
        Default to the unified schema used across agents (flat dict; no 'extras').
        """
        return '{"text": str?, "images": [path]?, "cifs": [path]?, "bbox": {x,y,w,h}?}'

    def _fill_from_history(self, obj: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Fill missing keys from recent internal memory payloads if possible."""
        used = False
        # scan recent payloads/results for dicts to backfill missing fields
        needed_keys = []
        try:
            # parse keys from schema text heuristically
            schema = self.expected_input_schema()
            for tok in [t.strip(' {}\"') for t in schema.replace(':', ',').split(',')]:
                if tok and all(c.isalnum() or c in ('_',) for c in tok) and tok not in obj:
                    needed_keys.append(tok)
        except Exception:
            pass
        if not needed_keys:
            return obj, used
        for e in reversed(self.memory[-20:]):
            for src in (e.get('payload'), e.get('result')):
                if isinstance(src, dict):
                    for k in needed_keys:
                        if k not in obj and k in src:
                            obj[k] = src[k]
                            used = True
            if all(k in obj for k in needed_keys):
                break
        return obj, used

    def normalize_input(self, raw: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Use LLM (if available) to validate and normalize agent input to JSON string
        following expected_input_schema(). Missing fields may be filled from
        internal memory when sensible. Returns (json_str, meta).
        """
        meta: Dict[str, Any] = {"validated": False, "used_history": False}
        # If already dict, start there; if string try JSON; else wrap into {"query": str}
        obj: Dict[str, Any]
        if isinstance(raw, dict):
            obj = dict(raw)
        else:
            s = raw if isinstance(raw, str) else str(raw)
            try:
                obj = json.loads(s)
            except Exception:
                # best-effort wrap
                obj = {"query": s}
        # Backfill from memory before LLM
        obj, used_hist = self._fill_from_history(obj)
        meta["used_history"] = bool(used_hist)

        # If no guard, return as-is
        if not self._guard:
            return json.dumps(obj, ensure_ascii=False), meta

        schema = self.expected_input_schema()
        history_txt = self.recent_memory_text(8)
        prompt = (
            "You validate and normalize inputs for a specific agent.\n"
            f"Agent: {self.name}.\n"
            f"Expected JSON schema (keys; '?' means optional): {schema}.\n"
            "Rules: Return only a single JSON object string.\n"
            "- If the input is already valid JSON matching the schema, echo a cleaned JSON.\n"
            "- If fields are missing but can be inferred from context/history, fill them.\n"
            "- If impossible, return {\"error\": \"...\"} describing what's wrong.\n"
            f"History (recent):\n{history_txt}\n"
            f"Input:\n{json.dumps(obj, ensure_ascii=False)}\n"
            "Output JSON:"
        )
        try:
            out = self._guard.forward(messages=[{"role": "system", "content": prompt}])
            # Attempt to extract JSON
            text = out.strip()
            # Guard against accidental framing text
            start = text.find('{'); end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
            norm = json.loads(text)
            meta["validated"] = True
            return json.dumps(norm, ensure_ascii=False), meta
        except Exception as e:
            # Fall back to original object; include error in history
            return json.dumps(obj, ensure_ascii=False), meta

    def normalize_agent_payload(self, raw: Any) -> Dict[str, Any]:
        """Return a unified payload with flat keys: text, images, cifs, bbox, ... (no 'extras').
        Uses emseek.protocol.normalize_agent_input and records the result.
        """
        try:
            from emseek.protocol import normalize_agent_input
            payload = normalize_agent_input(raw)
        except Exception:
            if isinstance(raw, dict):
                payload = {
                    "text": raw.get("text") or raw.get("query"),
                    "images": raw.get("images") or (raw.get("image_path") and [raw.get("image_path")]) or None,
                    "cifs": raw.get("cifs") or (raw.get("cif_path") and [raw.get("cif_path")]) or None,
                    "bbox": raw.get("bbox"),
                }
                # Pass through any additional keys directly (no 'extras')
                for k, v in raw.items():
                    if k not in {"text","query","images","image_path","image","cifs","cif_path","cif","bbox"}:
                        payload[k] = v
            else:
                payload = {"text": str(raw), "images": None, "cifs": None, "bbox": None}
        return payload

    def send_message(self, target_name, text=None, image=None):
        """
        Send a multimodal message to the target agent.
        Both text and image data are optional.
        """
        self.platform.send_message(self, target_name, text, image)

    def receive_message(self, sender, text=None, image=None):
        """
        Receive a multimodal message from another agent.
        The received message (text and/or image) is logged and printed.
        """
        info = f"Received message from {sender.name}"
        if text:
            info += f" | Text: {text}"
        if image:
            info += f" | Image: {image}"
        print(f"[{self.name}] {info}")
        # Record incoming text as-is; avoid normalize_input
        normalized = None
        parsed = None
        if text:
            try:
                obj = json.loads(text) if isinstance(text, str) else None
                parsed = obj if isinstance(obj, dict) else None
            except Exception:
                parsed = None
        self.platform.record_history(self.name, info, history={"parsed": parsed, "image": image})

    def perform_task(self, task, *args, **kwargs):
        """
        Perform a task and record its start and finish times.
        The task function is executed with the provided arguments.
        """
        start_info = f"Started task {task.__name__} at {time.strftime('%H:%M:%S')}"
        self.platform.record_history(self.name, start_info)
        result = task(*args, **kwargs)
        end_info = f"Finished task {task.__name__} at {time.strftime('%H:%M:%S')}, result: {result}"
        self.platform.record_history(self.name, end_info)
        return result

    # -------------------- Safe LLM wrappers --------------------
    def llm_call(self, messages=None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
        """Wrapper around self.llm.forward with error capture into history."""
        if not getattr(self, 'llm', None):
            return ""
        try:
            return self.llm.forward(messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)  # type: ignore[attr-defined]
        except Exception as e:
            return f"[ERROR] LLM call failed: {e}"

    def mllm_call(self, messages=None, image_path=None, image_url=None, image_b64=None,
                  max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
        """Wrapper around self.mllm.forward with error capture into history."""
        if not getattr(self, 'mllm', None):
            # Some agents keep a separate mllm_caller; try to use it if present
            caller = getattr(self, 'mllm_caller', None)
            if caller is None:
                return ""
            try:
                return caller.forward(messages=messages, image_path=image_path, image_url=image_url, image_b64=image_b64,
                                      max_tokens=max_tokens, temperature=temperature, **kwargs)
            except Exception as e:
                return f"[ERROR] MLLM call failed: {e}"
        try:
            return self.mllm.forward(messages=messages, image_path=image_path, image_url=image_url, image_b64=image_b64,
                                     max_tokens=max_tokens, temperature=temperature, **kwargs)  # type: ignore[attr-defined]
        except Exception as e:
            return f"[ERROR] MLLM call failed: {e}"

    # -------------------- Utilities --------------------
    @contextmanager
    def _suppress_errors(self):
        """Minimal context manager to ignore exceptions within a block.
        Keeps legacy call sites working without altering surrounding logic.
        """
        try:
            yield
        except Exception:
            pass
