import json
import re
from typing import Any, Dict, List, Tuple

try:
    from emseek.agents.base import Agent
except Exception:
    class Agent:  # minimal fallback
        def __init__(self, name, platform):
            self.name = name
            self.platform = platform


class GuardianAgent(Agent):
    """
    GuardianAgent (refactored):
    Curates the most useful information from recent multi‑agent history for the
    user's current query, then hands the curated context to ScribeAgent to compose
    the final answer.

    Input (unified): {text?: str, query?: str, goal?: str}
    Output (unified): {ok, message, text, images: null, meta}
    """

    def __init__(self, name, platform):
        super().__init__(name, platform)
        self.description = (
            "GuardianAgent (curator): gathers detailed facts/snippets from recent multi‑agent history "
            "for the current user query, preserving images and numeric details as much as possible.\n"
            "Does not describe or propose next‑step plans.\n"
            "Input: {text?: str, query?: str, goal?: str}\n"
            "Output: {ok, message, text, images: null, meta: {selected_indices, curated_context, agents_seen, sources, scribe?}}"
        )

    def expected_input_schema(self) -> str:
        return '{"text": str?, "query": str?, "goal": str?}'

    # -------------------- History collection --------------------
    def _history_items(self, k: int = 80) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        task_hist = self.platform.memory.get('task_memory', []) or []
        for entry in task_hist[-k:]:
            ts = entry.get('timestamp', '?')
            ag = entry.get('agent', '?')
            resp = entry.get('response')
            # Normalize to a rich text block (preserve details)
            snippet = ''
            if isinstance(resp, dict):
                msg = (resp.get('message') or '')
                txt = (resp.get('text') or '')
                parts: List[str] = []
                if isinstance(msg, str) and msg.strip():
                    parts.append("Message:")
                    parts.append(msg.strip())
                if isinstance(txt, str) and txt.strip():
                    parts.append("Text:")
                    parts.append(txt.strip())

                # Add image indicators
                imgs = resp.get('images') or resp.get('ref_images') or None
                try:
                    if imgs is not None:
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
                                        pth = im.get('path') or ''
                                        img_summ.append(f"  - [{i}] {kind} {pth} {('— ' + cap) if cap else ''}")
                                elif isinstance(im, str):
                                    img_summ.append(f"  - [{i}] string ({'base64' if len(im) > 64 else 'path'})")
                            if img_summ:
                                parts.append("Images:")
                                parts.extend(img_summ)
                        else:
                            parts.append("Images: 1 item")
                except Exception:
                    pass

                snippet = "\n".join(parts).strip()
            elif resp is not None:
                snippet = str(resp)
            # Keep newlines; do not aggressively truncate
            snippet = snippet.strip()
            items.append({
                'timestamp': ts,
                'agent': ag,
                'text': snippet,
                'raw': resp,
            })
        return items

    def _extract_sources(self) -> List[Any]:
        sources: List[Any] = []
        for entry in (self.platform.memory.get('task_memory') or [])[-80:]:
            resp = entry.get('response')
            if isinstance(resp, dict):
                meta = resp.get('meta') or {}
                for key in ('sources', 'refs', 'citations', 'provenance'):
                    val = meta.get(key)
                    if isinstance(val, list) and val:
                        sources.extend(val[:10])
        # Dedup
        seen = set(); uniq: List[Any] = []
        for s in sources:
            try:
                key = json.dumps(s, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = str(s)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    # -------------------- Relevance selection --------------------
    def _select_indices_llm(self, question: str, items: List[Dict[str, Any]], topk: int = 8) -> Tuple[List[int], str]:
        if not getattr(self, 'llm', None) or not items:
            return [], ''
        numbered = "\n".join([f"[{i}] ({it['agent']}) {it['text']}" for i, it in enumerate(items)])
        prompt = (
            "Select up to N most relevant items for answering the user question.\n"
            "Return JSON {\"indices\": [int...], \"notes\": str}.\n\n"
            f"Question: {question}\n"
            f"Items:\n{numbered}\n\n"
            f"N={topk}. Output JSON only:"
        )
        try:
            out = self.llm_call(messages=[{"role": "system", "content": prompt}])
            s = (out or '').strip(); i, j = s.find('{'), s.rfind('}')
            if i != -1 and j != -1 and j > i:
                data = json.loads(s[i:j+1])
                idxs = [int(x) for x in (data.get('indices') or []) if isinstance(x, (int, float))]
                idxs = [i for i in idxs if 0 <= i < len(items)][:topk]
                notes = str(data.get('notes') or '')
                return idxs, notes
        except Exception:
            pass
        return [], ''

    def _select_indices_heuristic(self, question: str, items: List[Dict[str, Any]], topk: int = 8) -> List[int]:
        q = re.findall(r"[A-Za-z0-9]+", (question or '').lower())
        qset = set(q)
        scores: List[Tuple[float, int]] = []
        for i, it in enumerate(items):
            t = (it.get('text') or '').lower()
            toks = set(re.findall(r"[A-Za-z0-9]+", t))
            overlap = len(qset & toks)
            # small bonus for known analysis agents
            ag = str(it.get('agent') or '').lower()
            bonus = 1 if any(k in ag for k in ['segmentor', 'analyzer', 'scholar', 'crystal', 'matprophet']) else 0
            length = min(len(t) / 200.0, 2.0)  # avoid overly long dominating
            score = overlap + bonus + length * 0.2
            scores.append((score, i))
        scores.sort(reverse=True)
        return [i for score, i in scores[:topk] if score > 0]

    # -------------------- Main entry --------------------
    def forward(self, messages: str | dict) -> Dict[str, Any]:
        # tolerant parse
        if isinstance(messages, dict):
            raw = dict(messages)
        else:
            s = messages if isinstance(messages, str) else str(messages)
            try:
                obj = json.loads(s)
                raw = obj if isinstance(obj, dict) else {"text": s}
            except Exception:
                raw = {"text": s}

        # unify question
        uni = self.normalize_agent_payload(raw)
        question = (uni.get('text') or raw.get('query') or raw.get('goal') or '').strip()

        items = self._history_items(80)
        agents_seen = list({(it.get('agent') or '?') for it in items})
        sources = self._extract_sources()

        if not items:
            msg = 'No recent history to curate.'
            scribe_out = None
            try:
                scribe = self.platform.agents.get('ScribeAgent')
                if scribe:
                    scribe_out = scribe.forward({"text": question or "Summarize the current state."})
            except Exception:
                pass
            return {
                "ok": True,
                "message": msg,
                "text": (scribe_out or {}).get('message') if isinstance(scribe_out, dict) else msg,
                "images": None,
                "meta": {"selected_indices": [], "curated_context": "", "agents_seen": agents_seen, "sources": sources or None, "scribe": scribe_out},
            }

        # LLM selection first, then heuristic fallback/augment
        idxs, notes = self._select_indices_llm(question, items, topk=8)
        if not idxs:
            idxs = self._select_indices_heuristic(question, items, topk=8)

        curated_lines = []
        for i in sorted(set(idxs)):
            it = items[i]
            curated_lines.append(f"[{i}] {it['agent']}: {it['text']}")
        curated_context = "\n".join(curated_lines)

        short_msg = (
            f"Curated {len(curated_lines)} relevant items from history." if curated_lines else
            "No highly relevant items detected; using minimal context."
        )

        # Hand off to ScribeAgent for final composition (ScribeAgent also reads platform history)
        scribe_out = None
        try:
            scribe = self.platform.agents.get('ScribeAgent')
            if scribe:
                scribe_out = scribe.forward({"text": question or "Summarize the curated context."})
        except Exception:
            scribe_out = None

        out = {
            "ok": True,
            "message": short_msg,
            "text": (notes or short_msg).strip(),
            "images": None,
            "meta": {
                "question": question,
                "selected_indices": sorted(set(idxs)),
                "curated_context": curated_context,
                "agents_seen": agents_seen,
                "sources": sources or None,
                "scribe": scribe_out,
            },
        }

        try:
            self.remember("forward", payload={"question": question, "selected": out["meta"]["selected_indices"]}, result={"curated_len": len(curated_lines)})
        except Exception:
            pass

        return out
