import re
import os
import base64
import json
import time
from emseek.agents import Agent
from emseek.protocol import save_base64_image, is_base64_str
from emseek.utils.mllm_caller import MLLMCaller
from emseek.utils.input import sanitize_json_like, flatten_inputs_obj


## moved to emseek.utils.input: sanitize_json_like, flatten_inputs_obj


class MaestroAgent(Agent):
    def __init__(self, name, platform):
        """
        Initialize the MaestroAgent.
        """
        super().__init__(name, platform)
        # Keep an MLLM handle only for optional, separate image parsing.
        self.mllm = MLLMCaller(self.platform.config)
        self.prompt = None
        self.last_thought = ""
        self.description = (
            "Maestro Orchestration/Router Agent: selects the next agent or produces the final answer, and normalizes call formats (no direct analysis/calculation).\n"
            "Capabilities:\n"
            "- Choose the next suitable agent or provide a final answer based on global/local history;\n"
            "- Uses LLM only for routing/decision output. Images are referenced by file paths.\n"
            "- If image understanding is required, Maestro may first call an MLLM to extract a brief textual description, then inject that text into the LLM prompt.\n"
            "Input spec (one of):\n"
            "- Plain text question; or\n"
            "- JSON string/object using: {text?: str, images?: str|[str], cifs?: str|[str], bbox?: {x,y,w,h}, ...};\n"
            "  • images/cifs can be a single path or a list; omit or set null if none;\n"
            "  • Put agent-specific parameters as top-level keys (no nested extras).\n"
            "Output spec (strictly one of; exactly one directive line; no extra text):\n"
            "1) |Action|<agent_name>|<inputs>|<messages>\n"
            "   - <inputs>: JSON string of the unified dictionary;\n"
            "   - <messages>: concise instruction to the target agent (if inputs lacks text, mirror messages into text).\n"
            "2) |Final Answer|None|<answer_text>\n"
            "Note: Maestro only decides and orchestrates; it does not edit files directly or access the network.\n"
            "Quick JSON example input: {\"text\": \"Please segment and describe the image\", \"images\": [\"/path/a.png\"]}"
        )

    def expected_input_schema(self) -> str:
        return '{"text": str?, "images": [path]?}'

    def ingest_image(self, image_input):
        """
        Ingest an image (base64/data URL or file path), persist it, and return (text, base64_preview).
        """
        work_dir = os.path.join(self.platform.working_folder, 'artifacts')
        img_path = None
        b64 = None
        if isinstance(image_input, str) and (
            image_input.startswith('/') or image_input.startswith('./')
            or image_input.endswith(('.png', '.jpg', '.jpeg'))
        ):
            img_path = os.path.abspath(image_input)
            try:
                with open(img_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
            except Exception:
                b64 = None
        else:
            try:
                b64_data = image_input
                img_path = save_base64_image(b64_data, work_dir, 'input')
                with open(img_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
            except Exception:
                img_path, b64 = None, None

        if img_path:
            try:
                img_path = os.path.abspath(img_path)
            except Exception:
                pass
            with self._suppress_errors():
                self.remember_file(img_path, label='Uploaded Image')
            return 'Image ingested for decision making.', b64
        else:
            return 'Failed to ingest the uploaded image.', None

    def forward(self, query):
        """
        Accepts raw string or dict-like payload. Normalizes input for routing,
        composes a fresh prompt with recent history (no unified block in prompt),
        converts images to file paths, optionally parses images via MLLM to text,
        then calls the LLM once (LLM-only for the final decision/output).
        """
        raw = query

        # Normalize for downstream routing & memory (NOT for prompt).
        if isinstance(query, dict) and any(k in query for k in ("text", "images", "cifs", "bbox")):
            obj = dict(query)
            uni = flatten_inputs_obj(obj)
        else:
            if isinstance(query, str):
                try:
                    tmp = json.loads(query)
                    obj = tmp if isinstance(tmp, dict) else {"text": query}
                except Exception:
                    obj = {"text": query}
            else:
                obj = {"text": str(query)}
            uni = flatten_inputs_obj(obj)

        query_text = (
            uni.get('text') if isinstance(uni.get('text'), str)
            else (obj.get('query') or obj.get('text') or (str(raw) if raw is not None else ''))
        )

        # Convert any base64/data-URL images to paths; keep paths as-is
        image_paths = []
        imgs = uni.get('images')
        if isinstance(imgs, list) and imgs:
            artifacts_dir = os.path.join(self.platform.working_folder, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            for idx, cand in enumerate(imgs):
                if not isinstance(cand, str) or not cand:
                    continue
                if is_base64_str(cand):
                    try:
                        p = save_base64_image(cand, artifacts_dir, f'maestro_input_{idx}')
                        image_paths.append(os.path.abspath(p))
                    except Exception:
                        continue
                else:
                    try:
                        image_paths.append(os.path.abspath(cand))
                    except Exception:
                        image_paths.append(cand)

        agent_list = self.platform.agents

        # Use full per-session history for detailed context (no truncation)
        task_memory = self.platform.memory.get("task_memory", []) or []
        lines = []
        for entry in task_memory:
            ts = entry.get("timestamp", "?")
            ag = entry.get("agent", "?")
            resp = entry.get("response", "")
            if resp is None:
                continue
            lines.append(f"{ts} - {ag}: {resp}")
        history_text = "\n".join(lines)

        # Optionally parse images with MLLM to get text-only descriptions when needed
        image_parse_text = ""
        if self._needs_image_parsing(query_text, image_paths):
            image_parse_text = self._parse_images_to_text(image_paths, query_text)

        # ======= NEW: compose prompt WITHOUT unified block (images listed as paths) =======
        images_for_prompt = image_paths if image_paths else None
        self.prompt = self.compose_prompt(images_for_prompt, agent_list, query, history_text, image_parse_text=image_parse_text)

        # LLM-only call for final decision/output
        try:
            response = self.llm_call(messages=[{"role": "system", "content": self.prompt}])
        except Exception as e:
            response = f"[ERROR] LLM call failed: {e}"

        print('----------Response-----------')
        print(response)
        print('------------------------------')

        task, agent_name, inputs, messages = self.parse_response(response, query_text)
        return task, agent_name, inputs, messages

    def compose_prompt(self, images_list, agent_list, query, history_text: str = "", image_parse_text: str = ""):
        """
        Contract-first prompt. NO 'unified' section—only history, agents, images (as paths), optional parsed image text, and user query.
        """
        try:
            em_images_str = "\n".join([str(p) for p in images_list if isinstance(p, str)]) if isinstance(images_list, list) else ""
        except Exception:
            em_images_str = ""

        agent_blocks = []
        for agent_name, ag in agent_list.items():
            if agent_name in ['MaestroAgent', 'GuardianAgent', 'ScribeAgent']:
                continue
            desc = getattr(ag, 'description', agent_name)
            try:
                state = ag.recent_memory_text(5) if hasattr(ag, 'recent_memory_text') else ''
            except Exception:
                state = ''
            agent_blocks.append(f"{agent_name}: {desc}\n  Recent State:\n{state if state else '(no recent state)'}")
        agents_str = "\n\n".join(agent_blocks)

        em_block = em_images_str.strip() if em_images_str else "(none)"
        history_block = f"### HISTORY\n{history_text.strip()}\n\n" if history_text and history_text.strip() else ""
        image_parse_block = f"### Image Analysis (Parsed)\n{image_parse_text.strip()}\n\n" if image_parse_text and image_parse_text.strip() else ""

        prompt = (
            # Role & mission
            "You are **EMSeek**, which focuses on bridging electron microscopy and materials analysis with an autonomous agentic platform.\n"
            "Your task is to CONTROL existing Agents and, using the continuously updated HISTORY, make the best possible decision and response.\n\n"

            # Output contract
            "### OUTPUT CONTRACT\n"
            "[Thought]: History: <Summarizing HISTORY> | Decision: <why route OR why answer directly>\n"
            "|Action|<agent_name>|<inputs_json>|<messages>\n"
            "OR\n"
            "|Final Answer|None|<answer_text>\n\n"

            # Requirements a–g
            "### REQUIREMENTS\n"
            "a) First, correctly understand the HISTORY and clarify the user's question.\n"
            "b) Then summarize current run results and judge whether they already solve the user's problem.\n"
            "c) Provide the decision rationale: why you chose this Agent to execute, and state the task clearly.\n"
            "d) `messages` is the effective channel to communicate with other Agents — it must be written in **natural language (not JSON)** and should state concrete tasks and requirements.\n"
            "e) If the user asks who you are, ALWAYS answer: EMSeek, and briefly state the core function above.\n"
            "f) `<inputs_json>` MUST be a flat JSON object like {text?, images?, cifs?, bbox?, ...}; omit any keys that are not present.\n"
            "g) The directive line MUST be the ONLY line starting with `|` and MUST be the LAST line of the entire output.\n\n"
            "h) Do NOT embed base64 or data URLs in <inputs_json>. When referring to images, use file paths exactly as listed under 'EM Files'.\n\n"
            "i) In **HISTORY**, entries from **MaestroAgent** are your own decision logs. Analyze them **chronologically** (by time) to avoid repeating the same decision or repeatedly invoking the same Agent when inputs/state have not changed. If repetition is intentional, explicitly justify it in [Thought].\n\n"
            "j) You may only output **one single task per turn**. If you need to plan multiple tasks, describe your task plan in `[Thought]` but execute only the immediate next step as the final directive line.\n\n"
            "k) If the most recent Agent execution has **failed**, you must analyze the failure cause based on HISTORY and attempt a reasonable solution or recovery before repeating the same call.\n\n"

            # Agent list
            "### Agent List\n"
            f"{agents_str.strip()}\n\n"

            # EM files block (paths only)
            "### EM Files\n"
            f"{em_block}\n\n"

            # HISTORY (optional)
            f"{history_block}"

            # Optional pre-parsed image notes
            f"{image_parse_block}"
            
            "### User Query"
            f"{query}"

            # Final reminder
            "The final line of your output must be a SINGLE directive line starting with `|`.\n"
            "In **HISTORY**, entries from **MaestroAgent** are your own decision logs. Analyze them **chronologically** (by time) to avoid repeating the same decision or repeatedly invoking the same Agent when inputs/state have not changed. If repetition is intentional, explicitly justify it in [Thought].\n"
        )

        print('**********History**********')
        print(history_block)
        print('********************')

        return prompt

    # -------------------- Helper methods --------------------
    def _needs_image_parsing(self, query_text: str, image_paths) -> bool:
        """
        Heuristic: decide whether we should parse image content via MLLM first.
        Trigger if (a) there are images and (b) the query suggests visual understanding
        or no meaningful text is present.
        """
        try:
            if not image_paths:
                return False
            qt = (query_text or "").strip().lower()
            if not qt:
                return True
            keywords = [
                # English-only triggers
                "describe", "what is in", "recognize", "recognise", "detect", "segment",
                "ocr", "read text", "visual", "image content", "figure", "photo", "picture",
                "image", "analyze image", "analyze picture", "image segmentation", "object detection",
            ]
            return any(k in qt for k in keywords)
        except Exception:
            return False

    def _parse_images_to_text(self, image_paths, query_text: str) -> str:
        """
        Use MLLM to produce concise textual descriptions of images.
        Returns a combined text block. This does not affect the final caller: the
        main decision is still produced by a text-only LLM call.
        """
        if not image_paths or not getattr(self, 'mllm', None):
            return ""
        notes = []
        for i, p in enumerate(image_paths[:3]):  # limit to first 3 images for brevity
            try:
                prompt = (
                    "You are given an image related to electron microscopy/materials. "
                    "Briefly describe its key visual contents and any text present (OCR concise), "
                    "in 3-6 bullet points. Focus on features relevant to analysis/routing. "
                    f"User context: {query_text}\n"
                )
                txt = self.mllm_call(messages=[{"role": "user", "content": prompt}], image_path=p)
                if isinstance(txt, str) and txt.strip():
                    notes.append(f"[Image {i+1}] {os.path.basename(p)}\n{txt.strip()}")
            except Exception:
                continue
        return "\n\n".join(notes).strip()

    def parse_response(self, response, original_query):
        """
        Parse:
          [Thought]: <...>
          |Action|<agent_name>|<inputs_json>|<messages>
          or
          |Final Answer|None|<answer_text>
        """
        plan_match = re.search(r'^\[(?:Thought|Plan)\]:?\s*(.*)$', response, re.MULTILINE)
        self.last_thought = plan_match.group(1).strip() if plan_match else ""

        directive_line = None
        for line in response.splitlines():
            if line.strip().startswith("|"):
                directive_line = line.strip()
        if not directive_line:
            safe_text = (response or "").strip() or str(original_query)
            return "Final Answer", "None", None, safe_text

        content = directive_line.strip().strip("|")
        parts = [p.strip() for p in content.split("|")]
        if not parts:
            safe_text = (response or "").strip() or str(original_query)
            return "Final Answer", "None", None, safe_text

        head = parts[0].lower()

        if head.startswith("final answer"):
            agent_name = parts[1] if len(parts) > 1 else "None"
            if (agent_name or "").lower() != "none":
                agent_name = "None"
            answer_text = parts[2] if len(parts) > 2 else (response.strip() or str(original_query))
            return "Final Answer", agent_name, None, answer_text

        if head.startswith("action"):
            agent_name = parts[1] if len(parts) > 1 else ""
            inputs_str = parts[2] if len(parts) > 2 else ""
            msg_str    = parts[3] if len(parts) > 3 else ""

            if agent_name and isinstance(self.platform.agents, dict):
                known = set(self.platform.agents.keys())
                if agent_name not in known:
                    lower_map = {k.lower(): k for k in known}
                    if agent_name.lower() in lower_map:
                        agent_name = lower_map[agent_name.lower()]
                    else:
                        clarification = (
                            f"Unrecognized agent '{agent_name}'. Available: {sorted(list(known))[:8]}..."
                            " Please specify a valid agent."
                        )
                        return "Final Answer", "None", None, clarification

            inputs_obj = {}
            if inputs_str:
                fixed = sanitize_json_like(inputs_str)
                try:
                    parsed = json.loads(fixed)
                    inputs_obj = parsed if isinstance(parsed, dict) else {"query": str(inputs_str)}
                except Exception:
                    inputs_obj = {"query": str(inputs_str)}

            inputs_obj = flatten_inputs_obj(inputs_obj)
            if msg_str and "text" not in inputs_obj:
                inputs_obj["text"] = msg_str[:160]

            return "Action", agent_name or "", inputs_obj, msg_str or ""

        safe_text = (response or "").strip() or str(original_query)
        return "Final Answer", "None", None, safe_text

    def summary(self, query):
        task_memory = self.platform.memory.get("task_memory", [])
        if not task_memory:
            return "No historical data available to summarize."

        history_str = ""
        for entry in task_memory:
            timestamp = entry.get("timestamp", "Unknown Time")
            agent = entry.get("agent", "Unknown Agent")
            response_text = entry.get("response", "")
            history_str += f"{timestamp} - {agent}: {response_text}\n"

        summary_prompt = (
            "You are tasked with summarizing the current results and history to answer the user's query. "
            f"the user's query context: {query}\n\n"
            "Based on the following execution history, provide a concise and coherent summary that addresses "
            f"the history:{history_str}\n\n"
            "Summary:"
        )
        summary_response = self.llm_call(messages=[{"role": "system", "content": summary_prompt}])
        return summary_response
