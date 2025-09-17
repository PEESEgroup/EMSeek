import os
import re
import json
import time
import asyncio
import logging
import threading
import functools
import concurrent.futures
from time import perf_counter
from pathlib import Path
from contextlib import suppress
from typing import List, Tuple, Optional, Any, Dict, Set, NoReturn
from collections import Counter
from logging.handlers import RotatingFileHandler

import requests

try:
    from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError  # type: ignore
except Exception:  # pragma: no cover - litellm may not expose exceptions in minimal installs
    LiteLLMAPIConnectionError = None

from emseek.agents.base import Agent


# =============================== Helpers ===============================

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default

def _now_ts() -> str:
    return str(int(time.time()))


# ===================== Shared PaperQA Docs (per folder) =====================

# key: absolute pdf_folder -> {"docs": Docs|None, "known": set[str], "evt": threading.Event, "dir_mtime_ns": int}
_SHARED_DOCS_POOL: Dict[str, Dict[str, Any]] = {}
_SHARED_POOL_LOCK = threading.Lock()

_CORE_STOPWORDS = {
    "the", "and", "with", "from", "that", "into", "using", "for", "into", "this", "have", "been",
    "their", "your", "about", "there", "were", "which", "based", "study", "analysis", "data",
    "paper", "results", "effect", "effects", "approach", "model", "models", "method", "methods",
    "research", "investigation", "introduction", "conclusion", "review", "impact", "system", "systems",
    "role", "roles", "chemical", "materials", "material", "application", "applications", "performance",
    "properties", "structure", "structures", "nanostructure", "nanostructures", "behaviour", "behavior"
}


class ScholarSeekerUserError(RuntimeError):
    """Exception used to surface actionable errors to the caller."""

    def __init__(self, user_message: str, *, error_code: str = "user", internal_message: Optional[str] = None):
        super().__init__(internal_message or user_message)
        self.user_message = user_message
        self.error_code = error_code


# ====================================================================
#                      ScholarSeekerAgent (V2+)
# ====================================================================

class ScholarSeekerAgent(Agent):
    """
    ScholarSeekerAgent (V2+): PaperQA + CORE research agent with shared Docs,
    warmup, manifest-driven incremental preload, parallel CORE, and ordered downloads.

    Optional environment variables:
      - LLM_MODEL                (default: gpt-5-nano-2025-08-07)
      - CORE_API_KEY
      - CORE_TOPN                (default: 3)
      - MAX_CORE_ROUNDS          (default: 2)
      - USE_LLM_JUDGE            (default: 1)
      - LOG_LEVEL                (default: INFO)
      - LOG_TO_FILE              (default: 0)
      - LOG_JSON                 (default: 0)

    Performance knobs (optional):
      - PREWARM_PAPERQA          (default: 0)   # warm up Docs on __init__ (background)
      - AUTO_REFRESH_ON_QUERY    (default: 1)   # auto add new PDFs before query
      - PAPERQA_INIT_CONCURRENCY (default: 1)   # preload concurrency (aadd only)
      - CORE_SEARCH_CONCURRENCY  (default: 4)
      - CORE_DL_CONCURRENCY      (default: 4)
      - IO_WORKERS               (cap for thread pools)
    """

    # ------------------------------- init -------------------------------
    def __init__(self, name: str, platform: Any):
        super().__init__(name, platform)

        # Paths
        self.pdf_folder = Path(getattr(self.platform.config, "PDF_FOLDER", "database/papers_lib"))
        self.pdf_folder.mkdir(parents=True, exist_ok=True)
        self._shared_key = str(self.pdf_folder.resolve())
        self._dir_mtime_ns_local = 0  # local snapshot for change detection

        # Manifest (per folder)
        self._manifest_path = self.pdf_folder / ".paperqa_manifest.json"

        # Logging
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logger()

        # PaperQA docs cache & lock (agent-local async lock)
        self._docs_cache = None
        self._docs_lock = asyncio.Lock()

        # HTTP: thread-local pooled session (cuts TLS + handshake overhead)
        self._tls = threading.local()

        # Precompiled regex for arXiv normalization
        self._re_arxiv_abs = re.compile(r"https?://arxiv\.org/abs/([^?#]+)", re.I)
        self._re_arxiv_pdf = re.compile(r"https?://arxiv\.org/pdf/([^?#]+)$", re.I)

        # -------- Updated description (LLM controller aware)
        self.description = (
            "ScholarSeeker Literature QA Agent (LLM-controlled): validates inputs, prioritizes the local PDF library (PaperQA), "
            "and, if needed, searches CORE (with query improvement), normalizes arXiv/DOI to PDFs, then synthesizes a concise, sourced answer.\n"
            "Input spec (unified): {query?: str, text?: str, verbose?: bool, core_topn?: int, max_core_rounds?: int}\n"
            "Output spec (unified): {ok, message, text, images: null, meta}\n"
            "Notes: PDFs are saved into the local library; sources/diagnostics are included in meta."
        )

        # Optional pre-warm of PaperQA Docs (non-blocking by default)
        if (getattr(self.platform.config, "PREWARM_PAPERQA", None) if hasattr(self.platform, "config") else None) \
                if getattr(self.platform.config, "PREWARM_PAPERQA", None) is not None else _env_bool("PREWARM_PAPERQA", False):
            self.warmup(background=True)

    def expected_input_schema(self) -> str:
        # allow {text} as alias of {query}
        return '{"query": str?, "text": str?, "verbose": bool?, "core_topn": int?, "max_core_rounds": int?}'

    # ------------------------------ LLM controller ------------------------------

    def _error_dict(self, message: str, *, errors: Optional[List[str]] = None,
                    meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Unified failure dict for Maestro."""
        out = {
            "ok": False,
            "text": message,
            "message": message,
            "error": {"message": message, "fields": errors or []},
            "images": None,
            "cifs": None,
            "meta": meta or None,
        }
        with suppress(Exception):
            self.remember("error", payload={"errors": errors or [], "meta": meta}, result=message)
        return out

    def controller_validate_and_fix(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate/repair inputs for ScholarSeeker:
          - Accept {query} or {text} as the research question.
          - If empty, try LLM to infer from recent history.
        Returns: {ok, payload|None, errors, message}
        """
        errors: List[str] = []
        msg_parts: List[str] = []

        query = (payload.get("query") or payload.get("text") or "").strip()

        if not query:
            # Try to infer from history via LLM
            hist = self.recent_memory_text(6)
            if getattr(self, 'llm', None):
                try:
                    prompt = (
                        "You infer a single user research question from the agent's recent history if present.\n"
                        "Return JSON {\"query\": \"\" or string}. Use empty string if none.\n\n"
                        f"History:\n{hist}\n\nOutput JSON:"
                    )
                    out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                    s = (out or "").strip()
                    i, j = s.find("{"), s.rfind("}")
                    if i != -1 and j != -1 and j > i:
                        data = json.loads(s[i:j+1])
                        qq = data.get("query")
                        if isinstance(qq, str) and qq.strip():
                            query = qq.strip()
                            msg_parts.append("query inferred from history")
                except Exception:
                    pass

        if not query:
            errors.append("query")
            return {"ok": False, "payload": None, "errors": errors, "message": "Please provide a research question (query)."}

        fixed = dict(payload)
        fixed["query"] = query
        return {"ok": True, "payload": fixed, "errors": [], "message": "; ".join(msg_parts) or "ok"}

    def controller_summarize_success(self, *, query: str, answer_len: int, sources_cnt: int) -> str:
        """
        Produce a short one-liner (<=160 chars) for Maestro after success.
        """
        try:
            if getattr(self, 'llm', None):
                prompt = (
                    "Write one short sentence (<=160 chars) summarizing a successful literature QA.\n"
                    f"query_len={len(query)}, answer_len={answer_len}, sources={sources_cnt}. No extra commentary."
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                return (out or "").strip().splitlines()[0][:160]
        except Exception:
            pass
        # Fallback
        return f"Literature answer synthesized (sources={sources_cnt}, ~{answer_len} chars)."

    # ------------------------------ decision (kept for compatibility) ------------------------------
    def _decide_continue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        q = (payload.get('query') or payload.get('question') or '').strip()
        if q:
            return {"continue": True, "message": "ok"}
        # Try to infer from recent history
        hist = self.recent_memory_text(6)
        inferred = None
        if getattr(self, 'llm', None):
            try:
                prompt = (
                    "You infer a single user research question from agent's recent history if present.\n"
                    "Return JSON {\"query\": \"\" or string}. Use empty string if none.\n\n"
                    f"History:\n{hist}\n\nOutput JSON:"
                )
                out = self.llm_call(messages=[{"role": "system", "content": prompt}])
                s = out.strip(); i = s.find('{'); j = s.rfind('}')
                if i != -1 and j != -1 and j > i:
                    data = json.loads(s[i:j+1])
                    qq = data.get('query')
                    if isinstance(qq, str) and qq.strip():
                        inferred = qq.strip()
            except Exception:
                pass
        if inferred:
            return {"continue": True, "message": "inferred", "query": inferred}
        # Otherwise ask user for question
        return {"continue": False, "message": "Please provide a research question (query)."}

    # ------------------------------ logging -----------------------------
    def _setup_logger(self):
        if getattr(self, "_logger_inited", False):
            return

        level_name = str(
            getattr(self.platform.config, "LOG_LEVEL", None) or os.getenv("LOG_LEVEL", "INFO")
        ).upper()
        level = getattr(logging, level_name, logging.INFO)
        self.log.setLevel(level)
        self.log.propagate = False

        fmt_plain = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"

        use_json = (
            bool(getattr(self.platform.config, "LOG_JSON", False))
            if getattr(self.platform.config, "LOG_JSON", None) is not None else _env_bool("LOG_JSON", False)
        )
        log_to_file = (
            bool(getattr(self.platform.config, "LOG_TO_FILE", False))
            if getattr(self.platform.config, "LOG_TO_FILE", None) is not None else _env_bool("LOG_TO_FILE", False)
        )

        if not any(isinstance(h, logging.StreamHandler) for h in self.log.handlers):
            try:
                from rich.logging import RichHandler  # type: ignore
                ch = RichHandler(markup=False, show_path=False, rich_tracebacks=True)
                ch.setLevel(level)
                ch.setFormatter(logging.Formatter("%(message)s", datefmt=datefmt))
            except Exception:
                ch = logging.StreamHandler()
                ch.setLevel(level)
                ch.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))
            self.log.addHandler(ch)

        if log_to_file and not any(isinstance(h, RotatingFileHandler) for h in self.log.handlers):
            fh = RotatingFileHandler("emseek.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))
            self.log.addHandler(fh)

        self._logger_inited = True

    def _kv(self, **kwargs) -> str:
        if (bool(getattr(self.platform.config, "LOG_JSON", False))
                if getattr(self.platform.config, "LOG_JSON", None) is not None else _env_bool("LOG_JSON", False)):
            with suppress(Exception):
                return json.dumps(kwargs, ensure_ascii=False)
        parts = []
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)[:300]
            parts.append(f"{k}={v}")
        return " ".join(parts)

    def _section(self, title: str):
        self.log.info("-" * 60)
        self.log.info(f"> {title}")
        self.log.info("-" * 60)

    # ---------------------------- HTTP utils ----------------------------
    def _configure_session(self, sess: requests.Session):
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        sess.headers.update({"User-Agent": "ScholarSeekerAgent/1.0 (+https://example.local)"})

    def _get_session(self) -> requests.Session:
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = requests.Session()
            self._configure_session(sess)
            self._tls.session = sess
        return sess

    def _io_workers(self) -> int:
        cap = (
            int(getattr(self.platform.config, "IO_WORKERS", 0))
            if getattr(self.platform.config, "IO_WORKERS", None) is not None else _env_int("IO_WORKERS", 0)
        )
        if cap > 0:
            return cap
        return max(2, min(16, (os.cpu_count() or 4) * 2))

    def _search_workers(self) -> int:
        v = (
            int(getattr(self.platform.config, "CORE_SEARCH_CONCURRENCY", 0))
            if getattr(self.platform.config, "CORE_SEARCH_CONCURRENCY", None) is not None else _env_int("CORE_SEARCH_CONCURRENCY", 0)
        )
        return v if v > 0 else min(self._io_workers(), 4)

    def _dl_workers(self) -> int:
        v = (
            int(getattr(self.platform.config, "CORE_DL_CONCURRENCY", 0))
            if getattr(self.platform.config, "CORE_DL_CONCURRENCY", None) is not None else _env_int("CORE_DL_CONCURRENCY", 0)
        )
        return v if v > 0 else min(self._io_workers(), 4)

    # ---------------------------- Manifest utils ----------------------------
    def _manifest_load(self) -> Dict[str, Dict[str, int]]:
        try:
            if self._manifest_path.is_file():
                with self._manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _manifest_save(self, manifest: Dict[str, Dict[str, int]]) -> None:
        try:
            tmp = self._manifest_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False)
            tmp.replace(self._manifest_path)
        except Exception:
            self.log.exception("Manifest save failed")

    def _file_sig(self, p: Path) -> Tuple[int, int]:
        st = p.stat()
        return (st.st_size, st.st_mtime_ns)

    # ---------------------------- PaperQA warmup ----------------------------
    def warmup(self, background: bool = True):
        """
        Pre-warm PaperQA Docs (optionally in background). Does not change external semantics.
        """
        def _target():
            try:
                asyncio.run(self._load_docs())
            except Exception:
                self.log.exception("PaperQA warmup failed")
        if background:
            t = threading.Thread(target=_target, daemon=True, name="paperqa-warmup")
            t.start()
            return t
        else:
            return self._run_async(self._load_docs())

    async def _aadd_many_concurrent(self, docs, files: List[Path], concurrency: int) -> int:
        """
        Concurrent preload using Docs.aadd(...) with bounded concurrency.
        Default concurrency is 1 (sequential). Enable via PAPERQA_INIT_CONCURRENCY.
        """
        if concurrency <= 1 or not files:
            cnt = 0
            for p in files:
                with suppress(Exception):
                    await docs.aadd(p)
                    cnt += 1
            return cnt

        sem = asyncio.Semaphore(concurrency)
        async def worker(p: Path) -> bool:
            async with sem:
                with suppress(Exception):
                    await docs.aadd(p)
                    return True
            return False

        tasks = [asyncio.create_task(worker(p)) for p in files]
        added = 0
        for coro in asyncio.as_completed(tasks):
            try:
                if await coro:
                    added += 1
            except Exception:
                pass
        return added

    async def _preload_with_manifest(self, docs) -> int:
        """
        Manifest-aware preload:
        - Load manifest (path -> {size, mtime_ns})
        - Add only new/changed PDFs
        - Update manifest afterwards
        Returns number of added PDFs.
        """
        manifest = self._manifest_load()

        files = sorted(self.pdf_folder.glob("*.pdf"))
        to_add: List[Path] = []
        for p in files:
            k = str(p)
            sig = self._file_sig(p)
            rec = manifest.get(k)
            if (not rec) or tuple(rec.get("sig", (0, 0))) != sig:
                to_add.append(p)

        if not to_add:
            # nothing to add, but keep shared pool's known set in sync
            with _SHARED_POOL_LOCK:
                entry = _SHARED_DOCS_POOL.get(self._shared_key)
                if entry:
                    entry["known"] = set(str(p) for p in files)
            return 0

        aadd = getattr(docs, "aadd", None)
        added = 0
        if callable(aadd):
            cc = max(
                1,
                (
                    int(getattr(self.platform.config, "PAPERQA_INIT_CONCURRENCY", 1))
                    if getattr(self.platform.config, "PAPERQA_INIT_CONCURRENCY", None) is not None
                    else _env_int("PAPERQA_INIT_CONCURRENCY", 1)
                ),
            )
            added = await self._aadd_many_concurrent(docs, to_add, cc)
        else:
            add = getattr(docs, "add", None)
            if callable(add):
                for p in to_add:
                    with suppress(Exception):
                        add(p)
                        added += 1

        # update manifest
        for p in to_add:
            manifest[str(p)] = {"sig": self._file_sig(p)}
        self._manifest_save(manifest)

        # sync shared "known"
        with _SHARED_POOL_LOCK:
            entry = _SHARED_DOCS_POOL.get(self._shared_key)
            if entry:
                entry.setdefault("known", set()).update(str(p) for p in to_add)

        return added

    async def _load_docs(self):
        """
        Get or initialize a shared Docs and preload PDFs from the folder.
        Uses manifest to skip unchanged files; optional concurrency if aadd exists.
        """
        try:
            from paperqa import Docs
        except ModuleNotFoundError as e:  # pragma: no cover - dependent on optional install
            msg = (
                "PaperQA is not installed. Install it with `pip install paper-qa` or disable ScholarSeeker."
            )
            self.log.error(msg)
            raise ScholarSeekerUserError(
                msg,
                error_code="paperqa_missing",
                internal_message=f"PaperQA module missing: {e}"
            ) from e

        key = self._shared_key

        # Fast path: shared docs already initialized
        with _SHARED_POOL_LOCK:
            entry = _SHARED_DOCS_POOL.get(key)
            if entry and entry.get("docs") is not None:
                self._docs_cache = entry["docs"]
                return self._docs_cache

            # Prepare init or wait event
            if entry is None:
                entry = {"docs": None, "known": set(), "evt": threading.Event(), "dir_mtime_ns": 0}
                _SHARED_DOCS_POOL[key] = entry
                init_this_caller = True
                evt = entry["evt"]
            else:
                evt = entry.get("evt")
                if evt is None or evt.is_set():
                    evt = threading.Event()
                    entry["evt"] = evt
                    init_this_caller = True
                else:
                    init_this_caller = False

        # If another thread is initializing, wait without blocking the loop
        if not init_this_caller:
            await asyncio.to_thread(evt.wait)
            with _SHARED_POOL_LOCK:
                self._docs_cache = _SHARED_DOCS_POOL[key]["docs"]
            return self._docs_cache

        # Initialize once
        t0 = perf_counter()
        self._section("Initialize PaperQA Docs & preload local PDFs (shared)")
        docs = Docs()

        # Manifest-aware preload (with optional concurrency)
        added = await self._preload_with_manifest(docs)

        # Commit to shared pool and release waiters
        with _SHARED_POOL_LOCK:
            entry = _SHARED_DOCS_POOL[key]
            entry["docs"] = docs
            entry["known"] = set(str(p) for p in sorted(self.pdf_folder.glob("*.pdf")))
            try:
                entry["dir_mtime_ns"] = self.pdf_folder.stat().st_mtime_ns
            except Exception:
                entry["dir_mtime_ns"] = 0
            entry["evt"].set()
            self._docs_cache = docs

        took = f"{perf_counter() - t0:.2f}s"
        self.log.info(
            f"Preloaded PDFs (shared) | "
            f"{self._kv(added=added, folder=str(self.pdf_folder), took=took)}"
        )
        return self._docs_cache

    async def _paperqa_ask(self, question: str, return_sources: bool = True) -> Tuple[str, List[str]]:
        docs = await self._load_docs()
        settings = self._paperqa_settings()
        t0 = perf_counter()
        aquery = getattr(docs, "aquery", None)
        try:
            if callable(aquery):
                session = await (aquery(question, settings=settings) if settings else aquery(question))
            else:
                session = docs.query(question, settings=settings) if settings else docs.query(question)
        except Exception as e:
            self._handle_paperqa_exception(e, stage="query")
        finally:
            self.log.info(f"PaperQA query done | {self._kv(took=f'{perf_counter()-t0:.2f}s')}")

        ans = getattr(session, "formatted_answer", None) or getattr(session, "answer", "") or str(session)
        if return_sources:
            ctx = getattr(session, "context", []) or []
            srcs = [str(c) for c in ctx]
            return ans or "", srcs
        return ans or "", []

    def _paperqa_settings(self):
        model = getattr(self.platform.config, "LLM_MODEL", None) or os.getenv("LLM_MODEL") or "gpt-5-nano-2025-08-07"
        try:
            from paperqa import Settings  # type: ignore
            self.log.debug(f"PaperQA Settings | {self._kv(llm=model, temperature=1.0)}")
            return Settings(lookup_paper_metadata=False, answer_max_sources=10, llm=model, temperature=1.0)
        except Exception:
            self.log.debug("PaperQA Settings unavailable; using defaults")
        return None

    async def _paperqa_add(self, files: List[str]):
        """
        Add new PDFs to existing Docs (incremental). Sequential by design to preserve stability.
        Also updates manifest and shared 'known'.
        """
        if not files:
            return
        docs = await self._load_docs()
        ok, fail = 0, 0
        aadd = getattr(docs, "aadd", None)
        if callable(aadd):
            for p in files:
                try:
                    await aadd(Path(p))
                    ok += 1
                except Exception:
                    fail += 1
                    self.log.exception(f"Add to Docs failed | {self._kv(path=p)}")
        else:
            add = getattr(docs, "add", None)
            if callable(add):
                for p in files:
                    try:
                        add(Path(p))
                        ok += 1
                    except Exception:
                        fail += 1
                        self.log.exception(f"Add to Docs failed | {self._kv(path=p)}")
        self.log.info(f"Docs updated | {self._kv(ok=ok, fail=fail)}")

        # Update manifest for these files
        manifest = self._manifest_load()
        for s in files:
            p = Path(s)
            try:
                manifest[str(p)] = {"sig": self._file_sig(p)}
            except Exception:
                pass
        self._manifest_save(manifest)

        # Update shared 'known' and dir mtime
        with _SHARED_POOL_LOCK:
            entry = _SHARED_DOCS_POOL.get(self._shared_key)
            if entry:
                entry.setdefault("known", set()).update(files)
                try:
                    entry["dir_mtime_ns"] = self.pdf_folder.stat().st_mtime_ns
                except Exception:
                    pass

    def _current_pdf_paths(self) -> List[Path]:
        return sorted(self.pdf_folder.glob("*.pdf"))

    async def _refresh_docs_if_needed(self) -> int:
        """
        If new PDFs are found in the folder, incrementally add them to Docs.
        Returns the number of newly added PDFs.
        """
        await self._load_docs()
        key = self._shared_key
        with _SHARED_POOL_LOCK:
            entry = _SHARED_DOCS_POOL.get(key)
            if not entry or entry.get("docs") is None:
                return 0
            last_mtime_ns = int(entry.get("dir_mtime_ns") or 0)

        try:
            curr_mtime_ns = self.pdf_folder.stat().st_mtime_ns
        except Exception:
            curr_mtime_ns = 0

        if curr_mtime_ns == last_mtime_ns and self._dir_mtime_ns_local == curr_mtime_ns:
            return 0

        files_now = self._current_pdf_paths()
        with _SHARED_POOL_LOCK:
            known = entry.get("known") or set()
            new_paths = [str(p) for p in files_now if str(p) not in known]

        if not new_paths:
            with _SHARED_POOL_LOCK:
                entry["dir_mtime_ns"] = curr_mtime_ns
            self._dir_mtime_ns_local = curr_mtime_ns
            return 0

        await self._paperqa_add(new_paths)
        with _SHARED_POOL_LOCK:
            entry["known"].update(new_paths)
            entry["dir_mtime_ns"] = curr_mtime_ns
        self._dir_mtime_ns_local = curr_mtime_ns
        return len(new_paths)

    # --------------------------- CORE integration ---------------------------
    def core_api_key(self) -> str:
        try:
            return getattr(self.platform.config, "CORE_API_KEY", "") or os.getenv("CORE_API_KEY", "")
        except Exception:
            return os.getenv("CORE_API_KEY", "")

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:120]

    def _download_pdf(self, url: str, title: str, stop_event: Optional[threading.Event] = None) -> Optional[str]:
        """Download a PDF with detailed logging. Supports cooperative cancellation."""
        if stop_event and stop_event.is_set():
            return None

        t0 = perf_counter()
        self.log.info(f"Download PDF | {self._kv(url=url, title=title)}")
        try:
            headers = {
                "User-Agent": "ScholarSeekerAgent/1.0 (+https://example.local)",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            }
            sess = self._get_session()
            with sess.get(url, timeout=30, headers=headers, stream=True) as r:
                if not r.ok:
                    self.log.warning(f"HTTP not OK | {self._kv(status=r.status_code, reason=r.reason)}")
                    return None

                ct = (r.headers.get("content-type") or "").lower()
                self.log.debug(f"Response headers | {self._kv(content_type=ct)}")

                fname = self._sanitize_filename(title or _now_ts()) + ".pdf"
                fpath = self.pdf_folder / fname

                total = 0
                aborted = False
                with open(fpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        if stop_event and stop_event.is_set():
                            aborted = True
                            break
                        f.write(chunk)
                        total += len(chunk)

                if aborted:
                    with suppress(Exception):
                        fpath.unlink(missing_ok=True)  # py>=3.8
                    self.log.info(f"Download aborted (early stop) | {self._kv(url=url, title=title)}")
                    return None

                self.log.info(f"Downloaded | {self._kv(path=str(fpath), bytes=total, took=f'{perf_counter()-t0:.2f}s')}")
                try:
                    self.remember_file(str(fpath), label='pdf', kind='paper')
                except Exception:
                    pass
                return str(fpath)
        except Exception:
            self.log.exception(f"Download failed | {self._kv(url=url)}")
            return None

    def _normalize_pdf_url(self, url: str, rec: Optional[dict] = None) -> str:
        """Normalize common non-direct links to direct PDFs (arXiv/DOI)."""
        if not url:
            return url
        u = url.strip()

        m = self._re_arxiv_abs.match(u)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}.pdf"

        m2 = self._re_arxiv_pdf.match(u)
        if m2 and not m2.group(1).lower().endswith(".pdf"):
            return f"https://arxiv.org/pdf/{m2.group(1)}.pdf"

        if u.lower().endswith(".pdf"):
            return u

        if "doi.org/" in u.lower():
            doi = u.split("doi.org/", 1)[1].split("?", 1)[0]
            pdf = self._resolve_doi_to_pdf(doi)
            return pdf or u

        if rec:
            try:
                for ident in (rec.get("identifiers") or []):
                    if str(ident.get("type", "")).upper() in {"ARXIV_ID", "ARXIV"}:
                        return f"https://arxiv.org/pdf/{ident.get('identifier')}.pdf"
            except Exception:
                pass
            try:
                for oid in rec.get("oaiIds") or []:
                    if isinstance(oid, str) and "arxiv.org" in oid:
                        arxid = oid.split(":")[-1]
                        return f"https://arxiv.org/pdf/{arxid}.pdf"
            except Exception:
                pass

        return u

    def _handle_paperqa_exception(self, err: Exception, stage: str) -> NoReturn:
        """Translate PaperQA/LLM failures into actionable errors for callers."""
        if isinstance(err, ScholarSeekerUserError):
            raise err

        message = str(err)
        lower = message.lower()

        if LiteLLMAPIConnectionError and isinstance(err, LiteLLMAPIConnectionError):
            user_msg = (
                "ScholarSeeker could not reach the language model service. "
                "Verify network access and OPENAI_API_KEY credentials."
            )
            self.log.error(f"PaperQA {stage} connection failure | {self._kv(error=message)}")
            raise ScholarSeekerUserError(
                user_msg,
                error_code="llm_connection",
                internal_message=message
            ) from err

        if "connection error" in lower or "failed to establish" in lower or "timed out" in lower:
            user_msg = (
                "ScholarSeeker's LLM request failed to connect. "
                "Check network connectivity and API credentials, then retry."
            )
            self.log.error(f"PaperQA {stage} connectivity problem | {self._kv(error=message)}")
            raise ScholarSeekerUserError(
                user_msg,
                error_code="llm_connection",
                internal_message=message
            ) from err

        if "api key" in lower or "unauthorized" in lower:
            user_msg = (
                "ScholarSeeker could not authenticate with the language model provider. "
                "Confirm that OPENAI_API_KEY is set and valid."
            )
            self.log.error(f"PaperQA {stage} auth failure | {self._kv(error=message)}")
            raise ScholarSeekerUserError(
                user_msg,
                error_code="llm_auth",
                internal_message=message
            ) from err

        self.log.exception(f"PaperQA {stage} unexpected failure")
        raise err

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text or "")
        seen: Set[str] = set()
        keywords: List[str] = []
        for tok in tokens:
            word = tok.strip('-').lower()
            if len(word) < 4:
                continue
            if word in _CORE_STOPWORDS:
                continue
            if word not in seen:
                seen.add(word)
                keywords.append(word)
        return keywords

    def _core_keywords_from_record(self, rec: Dict[str, Any]) -> List[str]:
        pieces: List[str] = []
        for key in ("title", "abstract", "description", "subtitle"):
            val = rec.get(key)
            if isinstance(val, str):
                pieces.append(val)

        for key in ("topics", "subjects", "fields", "fieldsOfStudy", "keywords"):
            val = rec.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                pieces.append(v)

        if not pieces:
            return []

        combined = " ".join(pieces)
        return self._extract_keywords_from_text(combined)

    @functools.lru_cache(maxsize=2048)
    def _resolve_doi_to_pdf(self, doi: str, timeout: int = 20) -> Optional[str]:
        """Use DOI resolver with Accept: application/pdf to reach a final PDF URL."""
        try:
            headers = {
                "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.5",
                "User-Agent": "ScholarSeekerAgent/1.0",
            }
            sess = self._get_session()
            r = sess.get(f"https://doi.org/{doi}", headers=headers, timeout=timeout, allow_redirects=True)
            if not r.ok:
                return None
            ct = (r.headers.get("content-type") or "").lower()
            if "pdf" in ct:
                return r.url
            if r.url.lower().endswith(".pdf"):
                return r.url
            return None
        except Exception:
            return None

    def _extract_pdf_url_and_title(self, rec: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Extract the most plausible PDF URL + title from a CORE record."""
        title = (rec.get("title") or rec.get("id") or _now_ts())
        candidates: List[str] = []

        for k in ("downloadUrl", "fullTextLink"):
            u = rec.get(k)
            if isinstance(u, str):
                candidates.append(u)

        ft = rec.get("fulltext") or {}
        if isinstance(ft, dict):
            u = ft.get("link")
            if isinstance(u, str):
                candidates.append(u)

        for u in rec.get("sourceFulltextUrls") or rec.get("fulltextUrls") or []:
            if isinstance(u, str):
                candidates.append(u)

        for lk in rec.get("links") or rec.get("fullTextLinks") or []:
            if isinstance(lk, dict):
                u = lk.get("url") or lk.get("href")
                if isinstance(u, str):
                    candidates.append(u)

        for u in candidates:
            u_norm = self._normalize_pdf_url(u, rec=rec)
            if isinstance(u_norm, str) and u_norm.strip():
                return u_norm, str(title)

        return None, str(title)

    # ------------------------- CORE fetch (parallel) -------------------------
    def _download_candidates_in_order(
        self,
        candidates: List[Tuple[str, str]],
        need: int,
        stop_event: threading.Event
    ) -> List[str]:
        """
        Ordered-concurrency downloader:
        - Spawn up to dl_workers tasks
        - Accept results strictly in input order
        - Stop early after collecting `need` successes
        """
        if need <= 0 or not candidates:
            return []

        dl_workers = min(self._dl_workers(), max(1, need))
        NOT_DONE = object()

        results: List[Any] = [NOT_DONE] * len(candidates)  # index -> str(path)|None|NOT_DONE
        next_to_schedule = 0
        next_to_commit = 0
        accepted: List[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=dl_workers, thread_name_prefix="core-dl") as ex:
            futures: Dict[concurrent.futures.Future, int] = {}

            # Initial fill
            while len(futures) < dl_workers and next_to_schedule < len(candidates):
                idx = next_to_schedule
                url, title = candidates[idx]
                fut = ex.submit(self._download_pdf, url, title, stop_event)
                futures[fut] = idx
                next_to_schedule += 1

            while futures and len(accepted) < need:
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    idx = futures.pop(fut)
                    try:
                        results[idx] = fut.result()
                    except Exception:
                        results[idx] = None

                # Commit strictly in order
                while next_to_commit < len(candidates):
                    val = results[next_to_commit]
                    if val is NOT_DONE:
                        break
                    if val:
                        accepted.append(val)
                        next_to_commit += 1
                        if len(accepted) >= need:
                            stop_event.set()
                            break
                    else:
                        next_to_commit += 1

                while len(futures) < dl_workers and next_to_schedule < len(candidates) and len(accepted) < need:
                    idx = next_to_schedule
                    url, title = candidates[idx]
                    fut = ex.submit(self._download_pdf, url, title, stop_event)
                    futures[fut] = idx
                    next_to_schedule += 1

        return accepted

    def _core_search_once(self, query: str, limit: int) -> List[dict]:
        key = self.core_api_key()
        if not key:
            self.log.warning("CORE_API_KEY not set; skipping CORE search")
            return []
        url = "https://api.core.ac.uk/v3/search/works"  # CORE v3 "works" endpoint
        params = {"q": query, "limit": limit}
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        try:
            sess = self._get_session()
            r = sess.get(url, params=params, headers=headers, timeout=20)
            if not r.ok:
                self.log.warning(f"CORE request not OK | {self._kv(status=r.status_code)}")
                return []
            data = r.json()
            return (data.get("results") or data.get("data") or [])
        except Exception:
            self.log.exception("CORE search failed")
            return []

    def _core_fetch_pdfs(self, queries: List[str], topn: int) -> Dict[str, Any]:
        """
        Search CORE with given queries, download up to topn PDFs (total), and
        collect lightweight feedback (frequent keywords, empty queries).
        """
        self._section("CORE search & download")
        files: List[str] = []
        feedback = {"files": files, "keywords": [], "empty_queries": []}
        if topn <= 0:
            return feedback

        search_workers = self._search_workers()
        q_order: List[str] = [q for q in queries if q]
        results_by_q: Dict[str, List[dict]] = {}
        kw_counter: Counter[str] = Counter()

        if not q_order:
            self.log.info("No CORE queries provided.")
            return feedback

        self.log.info(f"CORE multi-search | {self._kv(queries=len(q_order), workers=search_workers, per_limit=topn)}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=search_workers, thread_name_prefix="core-search") as ex:
            futs = {ex.submit(self._core_search_once, q, topn): q for q in q_order}
            # Collect in original order (preserves selection order)
            for q in q_order:
                fut = next(f for f, qq in futs.items() if qq == q)
                try:
                    results_by_q[q] = fut.result()
                except Exception:
                    self.log.exception(f"CORE search future failed | {self._kv(q=q)}")
                    results_by_q[q] = []

        stop_event = threading.Event()
        for q in q_order:
            if len(files) >= topn:
                break
            results = results_by_q.get(q) or []
            self.log.info(f"CORE query | {self._kv(q=q, limit=topn)}")
            self.log.info(f"CORE results | {self._kv(count=len(results))}")
            if not results:
                feedback["empty_queries"].append(q)

            candidates: List[Tuple[str, str]] = []
            for rec in results:
                pdf_url, title = self._extract_pdf_url_and_title(rec)
                kws = self._core_keywords_from_record(rec)
                if kws:
                    kw_counter.update(kws)
                if pdf_url:
                    self.log.info(f"Try PDF | {self._kv(title=title, url=pdf_url)}")
                    candidates.append((pdf_url, title))

            if not candidates:
                continue

            need = topn - len(files)
            got = self._download_candidates_in_order(candidates, need, stop_event)
            files.extend(got)

        self.log.info(f"CORE downloaded PDFs | {self._kv(count=len(files))}")
        if kw_counter:
            feedback["keywords"] = [kw for kw, _ in kw_counter.most_common(12)]
        return feedback

    # ------------------------------ LLM utils ------------------------------
    def _llm_call(self, prompt: str) -> str:
        """Use base agent's safe LLM call wrapper and capture errors into history."""
        out = self.llm_call(messages=[{"role": "user", "content": prompt}])
        return (out or "").strip()

    def _judge_sufficient(self, q: str, a: str) -> bool:
        """
        Strict sufficiency check: heuristic + optional LLM (YES/NO).
        """
        text = (a or "").strip()
        if len(text) < 80:
            h_ok = False
        else:
            bad = ["don't know", "do not know", "not sure", "no information", "N/A", "do not know", "cannot answer"]
            h_ok = not any(b in text.lower() for b in bad)

        if not (
            bool(getattr(self.platform.config, "USE_LLM_JUDGE", True))
            if getattr(self.platform.config, "USE_LLM_JUDGE", None) is not None
            else _env_bool("USE_LLM_JUDGE", True)
        ):
            return h_ok

        prompt = (
            "You are a strict reviewer. Judge if ASSISTANT_ANSWER sufficiently answers USER_QUESTION.\n"
            "Consider specificity, factuality, and directness. Respond with exactly one token: YES or NO.\n\n"
            f"USER_QUESTION:\n{q}\n\nASSISTANT_ANSWER:\n{a}\n\nAnswer:"
        )
        out = self._llm_call(prompt)
        if "YES" in out.upper():
            return True and h_ok
        if "NO" in out.upper():
            return False
        return h_ok

    def _improve_core_queries(
        self,
        question: str,
        last_answer: str = "",
        k: int = 3,
        extra_keywords: Optional[List[str]] = None,
        tried: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Ask LLM to craft up to k improved search queries for CORE.
        Optionally include keywords extracted from previous rounds and skip
        queries that were already attempted.
        """
        prompt = (
            "Given a user question about academic literature, craft up to "
            f"{k} improved search queries for CORE API.\n"
            "- Include key terms, synonyms, canonical names, acronyms.\n"
            "- Prefer English; if the question is non-English, include both a native and an English variant.\n"
            "- Use quotes for exact phrases only when beneficial.\n"
            "- Output plain queries, one per line, no extra text.\n\n"
            f"QUESTION:\n{question}\n\nOPTIONAL_LAST_ANSWER:\n{last_answer}\n"
        )
        if extra_keywords:
            prompt += "\nKNOWN_RELEVANT_TERMS:\n" + ", ".join(extra_keywords[:12]) + "\n"
        out = self._llm_call(prompt)
        lines = [re.sub(r'^[\-\d\.\)\s]+', '', s).strip() for s in (out or "").splitlines() if s.strip()]
        uniq: List[str] = []
        seen = tried or set()
        for s in lines:
            if not s:
                continue
            if s in seen:
                continue
            if s not in uniq:
                uniq.append(s)
            if len(uniq) >= k:
                break
        if uniq:
            return uniq

        fallback_candidates: List[str] = []
        toks = re.findall(r"[A-Za-z0-9\-]+", question)
        base = " ".join(toks[:8]) if toks else question
        if base:
            fallback_candidates.append(base)
        if question and question not in fallback_candidates:
            fallback_candidates.append(question)

        if extra_keywords:
            combined = " ".join(extra_keywords[:3]).strip()
            if combined:
                fallback_candidates.append(f"{combined} {base or question}".strip())
            for kw in extra_keywords[: max(3, k)]:
                cand = f"{kw} {base or question}".strip()
                fallback_candidates.append(cand)

        filtered: List[str] = []
        seen_out: Set[str] = set(seen)
        for cand in fallback_candidates:
            if not cand:
                continue
            if cand in seen_out:
                continue
            if cand not in filtered:
                filtered.append(cand)
            seen_out.add(cand)
            if len(filtered) >= k:
                break

        return filtered[:k]

    def _make_followups(self, question: str, last_answer: str, k: int = 3) -> List[str]:
        """
        Ask LLM for targeted follow-up sub-questions to deepen coverage.
        """
        prompt = (
            "Generate up to {k} short, targeted follow-up sub-questions that would help improve a literature answer.\n"
            "Focus on missing details, key metrics, comparisons, dates, or definitions.\n"
            "One per line. No numbering.\n\n"
            f"MAIN QUESTION:\n{question}\n\nCURRENT ANSWER:\n{last_answer}\n"
        ).replace("{k}", str(k))
        out = self._llm_call(prompt)
        lines = [re.sub(r'^[\-\d\.\)\s]+', '', s).strip() for s in (out or "").splitlines() if s.strip()]
        return lines[:k]

    def _synthesize_final(self, question: str, primary_answer: str, followup_pairs: List[Tuple[str, str]]) -> str:
        """
        Synthesize a concise, well-structured final answer using the provided answers.
        """
        parts = [f"QUESTION:\n{question}\n", f"PRIMARY_ANSWER:\n{primary_answer}\n", "FOLLOW_UPS:\n"]
        for fq, fa in followup_pairs:
            parts.append(f"- Q: {fq}\n  A: {fa}\n")
        prompt = (
            "Synthesize a concise, well-structured final answer to the QUESTION using the provided answers.\n"
            "If there are conflicts, resolve them conservatively and note uncertainties. Include key facts and caveats.\n"
            "Keep it under 250 words.\n\n" + "\n".join(parts)
        )
        return self._llm_call(prompt) or primary_answer

    # ------------------------------ public API ------------------------------
    def query(self, query: str, return_sources: bool = True, verbose: bool = False,
              core_topn: Optional[int] = None, max_core_rounds: Optional[int] = None):
        """
        Main entry: returns (final_answer, sources, debug_text).
        """
        if verbose:
            self.log.setLevel(logging.DEBUG)

        self._section("ScholarSeekerAgent query start")
        N = int(core_topn or getattr(self.platform.config, "CORE_TOPN", None) or os.getenv("CORE_TOPN", 3))
        R = int(max_core_rounds or getattr(self.platform.config, "MAX_CORE_ROUNDS", None) or os.getenv("MAX_CORE_ROUNDS", 2))
        self.log.info(self._kv(query=query, core_topn=N, max_core_rounds=R, pdf_lib=str(self.pdf_folder)))

        # Optional auto-refresh: pick up newly added PDFs before asking
        if (
            bool(getattr(self.platform.config, "AUTO_REFRESH_ON_QUERY", True))
            if getattr(self.platform.config, "AUTO_REFRESH_ON_QUERY", None) is not None
            else _env_bool("AUTO_REFRESH_ON_QUERY", True)
        ):
            try:
                added = self._run_async(self._refresh_docs_if_needed())
                if added:
                    self.log.info(f"Auto-refresh docs | {self._kv(new_pdfs=added)}")
            except Exception:
                self.log.exception("Auto-refresh docs failed")

        # 1) Ask PaperQA first
        ans, srcs = self._run_async(self._paperqa_ask(query, return_sources=True))
        last_answer, last_sources = ans, srcs
        self.log.info(f"Initial PaperQA | {self._kv(answer_len=len(last_answer), sources=len(last_sources))}")

        # 2) If insufficient, attempt CORE rounds (with query improvement)
        if not self._judge_sufficient(query, last_answer):
            core_queries: List[str] = [query]
            tried_queries: Set[str] = set(core_queries)
            feedback_terms: List[str] = []
            for round_id in range(R):
                self._section(f"CORE round {round_id + 1}")
                tried_queries.update(core_queries)
                feedback = self._core_fetch_pdfs(core_queries, topn=N)
                files = feedback.get("files") or []
                new_terms = feedback.get("keywords") or []
                empty_hits = feedback.get("empty_queries") or []
                for term in new_terms:
                    if term not in feedback_terms:
                        feedback_terms.append(term)
                if new_terms:
                    self.log.info(f"CORE feedback keywords | {self._kv(sample=new_terms[:5])}")
                if empty_hits:
                    self.log.debug(f"CORE queries without hits | {self._kv(queries=empty_hits)}")

                if files:
                    self._run_async(self._paperqa_add(files))
                    ans, srcs = self._run_async(self._paperqa_ask(query, return_sources=True))
                    last_answer, last_sources = ans or last_answer, (srcs or last_sources)
                    self.log.info(f"PaperQA after CORE | {self._kv(answer_len=len(last_answer), sources=len(last_sources))}")
                    if self._judge_sufficient(query, last_answer):
                        break

                core_queries = self._improve_core_queries(
                    query,
                    last_answer,
                    k=3,
                    extra_keywords=feedback_terms,
                    tried=tried_queries,
                )
                if not core_queries:
                    self.log.info("No further CORE query refinements suggested; stopping early.")
                    break

        # 3) If still insufficient, follow-ups then synthesize
        if not self._judge_sufficient(query, last_answer):
            self._section("Follow-up questioning")
            followups = self._make_followups(query, last_answer, k=3)
            follow_pairs: List[Tuple[str, str]] = []
            for fq in followups:
                fa, _ = self._run_async(self._paperqa_ask(fq, return_sources=False))
                follow_pairs.append((fq, fa))
            synthesized = self._synthesize_final(query, last_answer, follow_pairs)
            if synthesized:
                last_answer = synthesized

        debug = "\n".join([
            f"core_topn={N}",
            f"max_core_rounds={R}",
            *(f"source: {s}" for s in (last_sources or []))
        ]) if return_sources else f"core_topn={N} max_core_rounds={R}"

        return last_answer, (last_sources or []), debug

    # ------------------------------ runtime utils ------------------------------
    def _run_async(self, coro):
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(coro)

    # ------------------------------ unified forward ------------------------------
    def forward(self, payload: str | dict) -> Dict[str, Any]:
        """
        LLM-controlled entry. Accepts raw string or dict (unified).
        Returns:
          - success: {ok: True, message, text, images: None, meta}
          - failure: {ok: False, message, error{message, fields}, images: None}
        """
        # 0) tolerant parsing
        if isinstance(payload, dict):
            raw = dict(payload)
        else:
            s = str(payload)
            try:
                obj = json.loads(s)
                raw = obj if isinstance(obj, dict) else {"text": s}
            except Exception:
                raw = {"text": s}

        # 1) allow unified input {text, ...} to alias {query}
        try:
            uni = self.normalize_agent_payload(raw)
            if isinstance(uni.get('text'), str) and not raw.get('query'):
                raw['query'] = uni['text']
        except Exception:
            pass

        # 2) controller validate & repair
        ctrl = self.controller_validate_and_fix(raw)
        if not ctrl.get("ok", False):
            return self._error_dict(ctrl.get("message", "Invalid input."), errors=ctrl.get("errors", ["query"]))

        fixed = ctrl["payload"]
        q = fixed["query"]
        verbose = bool(fixed.get("verbose", False))
        core_topn = fixed.get("core_topn")
        max_core_rounds = fixed.get("max_core_rounds")

        # 3) run pipeline
        try:
            final_answer, sources, debug_text = self.query(
                q,
                return_sources=True,
                verbose=verbose,
                core_topn=core_topn,
                max_core_rounds=max_core_rounds,
            )
        except ScholarSeekerUserError as e:
            return self._error_dict(
                e.user_message,
                errors=[e.error_code],
                meta={"exception": str(e), "code": e.error_code}
            )
        except Exception as e:
            return self._error_dict("ScholarSeeker query failed.", errors=["internal"], meta={"exception": str(e)})

        # 4) short one-liner
        short_msg = self.controller_summarize_success(
            query=q, answer_len=len(final_answer or ""), sources_cnt=len(sources or [])
        )

        # 5) assemble output
        meta = {
            "sources": sources or [],
            "debug": debug_text,
            "core_topn": int(core_topn or getattr(self.platform.config, "CORE_TOPN", os.getenv("CORE_TOPN", 3))),
            "max_core_rounds": int(max_core_rounds or getattr(self.platform.config, "MAX_CORE_ROUNDS", os.getenv("MAX_CORE_ROUNDS", 2))),
        }

        with suppress(Exception):
            self.remember("forward", payload={"query": q}, result={"message": short_msg, "sources": len(sources or [])})

        return {
            "ok": True,
            "message": short_msg,
            "text": final_answer,
            "images": None,
            "meta": meta,
        }
