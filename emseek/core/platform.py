import time
import os
import json
import logging
from ..agents import (
    Agent,
    MaestroAgent,
    SegMentorAgent,
    ScholarSeekerAgent,
    AnalyzerHubAgent,
    CrystalForgeAgent,
    MatProphetAgent,
    GuardianAgent,
    ScribeAgent,
)
from .protocol import (
    normalize_mm_request,
    build_stream_step,
    build_stream_final,
    save_base64_image,
    path_to_base64_image,
    to_image_refs,
    is_base64_str,
)
import base64


class Platform:
    def __init__(self, config, working_folder=None, log_filename="platform.log"):
        """
        Initialize the Platform with a designated working folder.
        All logs and intermediate artifacts will be saved in this folder.

        Parameters:
            working_folder (str): The directory to save logs and artifacts.
            log_filename (str): The name of the log file.
        """
        self.config = config
        # Default to history-based folder if not provided
        if not working_folder:
            try:
                base = getattr(config, 'HISTORY_ROOT', 'history')
            except Exception:
                base = 'history'
            working_folder = os.path.join(base, 'default')
        # Normalize working folder to absolute path for stable artifact paths
        self.working_folder = os.path.abspath(working_folder)
        # Ensure the working folder exists; create subfolders for logs and artifacts.
        self.setup_working_folder()

        # Dictionary to store agent objects by name.
        self.agents = {}
        # Dictionary to store each agent's history log.
        # Each history entry is a dict with keys: timestamp, info, multimodal.
        self.memory = {
            'long_memory': [],
            'task_memory': [],
            'User': []
        }
        # Queue to store tasks: each task is a tuple (agent_name, task_function, args, kwargs)

        # Configure logging, placing the log file in the logs folder.
        log_path = os.path.join(self.working_folder, "logs", log_filename)
        self.logger = logging.getLogger("PlatformLogger")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info("Platform initialized with working folder: %s", self.working_folder)
        # Track current user/session for awareness
        self.current_user_id = None
        self.current_session_id = None

    def set_session(self, user_id: str = None, session_id: str = None):
        """
        Switch Platform context to a specific user/session so artifacts and
        logs are isolated. Safe to call per-request.
        """
        try:
            base = getattr(self.config, 'HISTORY_ROOT', 'history')
        except Exception:
            base = 'history'
        # Fallbacks
        u = user_id or self.current_user_id or 'default'
        s = session_id or self.current_session_id
        # Detect whether we're switching user/session
        prev_u = getattr(self, 'current_user_id', None)
        prev_s = getattr(self, 'current_session_id', None)
        switching = (u != prev_u) or (s != prev_s)

        # Build new working folder
        if s:
            new_folder = os.path.join(base, u, 'sessions', s)
        else:
            new_folder = os.path.join(base, u)
        # Update working folders
        self.setup_working_folder(new_folder)
        self.current_user_id = u
        self.current_session_id = s
        # Repoint FileHandler to the new logs path
        try:
            # Remove existing FileHandlers
            for h in list(self.logger.handlers):
                try:
                    if isinstance(h, logging.FileHandler):
                        self.logger.removeHandler(h)
                        try:
                            h.flush(); h.close()
                        except Exception:
                            pass
                except Exception:
                    pass
            log_path = os.path.join(self.working_folder, 'logs', 'platform.log')
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.info("Switched Platform session: user=%s session=%s folder=%s", u, s, self.working_folder)
        except Exception:
            pass
        # Only reset ephemeral task memory when switching to a different session/user
        if switching:
            try:
                self.memory['task_memory'] = []
            except Exception:
                pass

    def setup_working_folder(self, working_folder=None):
        """
        Create the working folder and its subfolders if they do not exist.
        Subfolders include 'logs' for log files and 'artifacts' for intermediate outputs.
        """
        if working_folder is not None:
            self.working_folder = os.path.abspath(working_folder)

        if not os.path.exists(self.working_folder):
            os.makedirs(self.working_folder)
        logs_folder = os.path.join(self.working_folder, "logs")
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        artifacts_folder = os.path.join(self.working_folder, "artifacts")
        if not os.path.exists(artifacts_folder):
            os.makedirs(artifacts_folder)

    def create_agent(self, name, agent_class=Agent):
        """
        Create a new agent of the specified class and initialize its history log.
        Raises an error if an agent with the same name exists.
        """
        if name in self.agents:
            self.logger.info(f"Agent '{name}' already exists!")
            return
        agent = agent_class(name, self)
        self.agents[name] = agent
        self.memory[name] = []
        self.logger.info("Created agent: %s", name)
        return agent

    def init_agent(self):
        # Orchestrator / manager
        self.create_agent('MaestroAgent', agent_class=MaestroAgent)
        # Core vision + retrieval + analysis agents
        self.create_agent('SegMentorAgent', agent_class=SegMentorAgent)
        self.create_agent('ScholarSeeker', agent_class=ScholarSeekerAgent)
        # Unified tool hub and specialized domain agents
        self.create_agent('AnalyzerHub', agent_class=AnalyzerHubAgent)
        self.create_agent('CrystalForge', agent_class=CrystalForgeAgent)
        self.create_agent('MatProphet', agent_class=MatProphetAgent)
        # Curator before final composition
        self.create_agent('GuardianAgent', agent_class=GuardianAgent)
        # Final composer
        self.create_agent('ScribeAgent', agent_class=ScribeAgent)

    def clear(self):
        self.memory['long_memory'] = []
        self.memory['task_memory'] = []
        self.memory['User'] = []
        self.agents['MaestroAgent'].clear()

    def record_history(self, agent_name, response=None, history=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Sanitize response: convert any base64/data-URL images to file paths
        def _ensure_artifacts_dir() -> str:
            p = os.path.join(self.working_folder, "artifacts")
            os.makedirs(p, exist_ok=True)
            return p

        def _b64_to_path(s: str) -> str:
            try:
                return save_base64_image(s, _ensure_artifacts_dir(), "history_img")
            except Exception:
                return s

        def _sanitize_images_value(val):
            # Handle common image collections: list of refs/strings or dict with base64
            try:
                if val is None:
                    return None
                # List of items
                if isinstance(val, list):
                    out = []
                    for it in val:
                        if isinstance(it, dict):
                            kind = (it.get('kind') or it.get('type') or '').lower()
                            if kind == 'base64' or 'data' in it or 'base64' in it:
                                data = it.get('data') or it.get('base64')
                                if isinstance(data, str) and is_base64_str(data):
                                    p = _b64_to_path(data)
                                    it2 = dict(it)
                                    it2.pop('data', None)
                                    it2.pop('base64', None)
                                    it2['kind'] = 'path'
                                    it2['type'] = 'path'
                                    it2['path'] = p
                                    out.append(it2)
                                else:
                                    out.append(it)
                            else:
                                out.append(it)
                        elif isinstance(it, str) and is_base64_str(it):
                            out.append(_b64_to_path(it))
                        else:
                            out.append(it)
                    return out
                # Single dict
                if isinstance(val, dict):
                    # If it itself looks like an image ref
                    kind = (val.get('kind') or val.get('type') or '').lower()
                    data = val.get('data') or val.get('base64')
                    if (kind == 'base64' or data) and isinstance(data, str) and is_base64_str(data):
                        p = _b64_to_path(data)
                        v2 = dict(val)
                        v2.pop('data', None)
                        v2.pop('base64', None)
                        v2['kind'] = 'path'
                        v2['type'] = 'path'
                        v2['path'] = p
                        return v2
                    # Else recurse fields
                    return {k: _sanitize_images_value(v) for k, v in val.items()}
                # Single string: if base64-like image, write to path
                if isinstance(val, str) and is_base64_str(val):
                    return _b64_to_path(val)
            except Exception:
                pass
            return val

        def _sanitize(obj):
            try:
                if obj is None:
                    return None
                # Dict: look for image-related keys and recurse
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        lk = str(k).lower()
                        if lk in ('images', 'ref_images', 'image', 'thumbnail', 'thumbnails', 'figures', 'plots'):
                            out[k] = _sanitize_images_value(v)
                        else:
                            out[k] = _sanitize(v)
                    return out
                # List/Tuple: sanitize each element
                if isinstance(obj, list):
                    return [_sanitize(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple(_sanitize(x) for x in obj)
            except Exception:
                pass
            return obj

        sanitized_response = _sanitize(response)

        # Per-agent memory: store the sanitized response; include optional plan/history for reference
        per_agent = {"timestamp": timestamp, "response": sanitized_response}
        if history is not None:
            per_agent["history"] = history
        self.memory.setdefault(agent_name, []).append(per_agent)

        # Global rolling memories
        log_entry = {"timestamp": timestamp, "agent": agent_name, "response": sanitized_response}
        if history is not None:
            log_entry["history"] = history
        self.memory['long_memory'].append(log_entry)
        self.memory['task_memory'].append(log_entry)

        # Persist to artifacts for debugging/traceability
        artifact_path = os.path.join(self.working_folder, "artifacts", "history.txt")
        try:
            with open(artifact_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | {agent_name} - Response: {response}\n")
        except Exception:
            pass
        # Persist JSONL log for programmatic analysis
        try:
            logs_dir = os.path.join(self.working_folder, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            jsonl_path = os.path.join(logs_dir, "history.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": timestamp,
                    "agent": agent_name,
                    "response": sanitized_response,
                    **({"history": history} if history is not None else {})
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def set_image(self, image=None, bbox=None, model=None):
        response_text, response_image = self.agents['SegMentor'].set_image(image, bbox, model)
        return response_text, response_image, None

    # --- Unified multimodal query API (files-based) --- #
    def query_unified(self, payload):
        """
        Unified Dict API that accepts text plus files (images/CIFs) as a single
        request and delegates to the standard query() pipeline.

        Input payload (recommended):
          {
            "user_id": "...",
            "text": "...",
            "files": [
               {"type":"image","data":"data:image/png;base64,..."} | 
               {"type":"cif","name":"foo.cif","text":"..."} |
               "/path/to/file.png" | {"path":"/path/to/file.cif"}
            ],
            "bbox": { ... },
            "options": { ... },
            "model": "general_model",   # independent top-level parameter
            "agent": "..."
          }

        Behavior:
          - Convert 'files' to internal path lists: images[], cifs[] (absolute paths if possible).
          - Build a unified agent payload {text, images, cifs, bbox, ...} (flat; no 'extras') and pass it directly to query().
          - Wrap query() output lines into step/final frames for the frontend.
        """
        data = payload or {}
        user_id = data.get('user_id')
        text = data.get('text')
        bbox = data.get('bbox')
        model = data.get('model')  # top-level model as independent param
        options = data.get('options') if isinstance(data.get('options'), dict) else None

        # Normalize 'files' from multiple legacy keys
        files = []
        if isinstance(data.get('files'), list):
            files.extend(data.get('files') or [])
        elif data.get('files') is not None:
            files.append(data.get('files'))
        # Back-compat: map single keys into files bucket
        for k in ('file', 'image', 'cif', 'images', 'cifs'):
            v = data.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                files.extend(v)
            else:
                files.append(v)

        # Persist files and split into images[] and cifs[] paths
        artifacts_dir = os.path.join(self.working_folder, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        image_paths = []
        cif_paths = []

        def _as_abs(p):
            try:
                return os.path.abspath(p)
            except Exception:
                return p

        # Helper: map API sample URL to local samples path
        def _map_api_samples(p: str) -> str:
            try:
                p = str(p)
            except Exception:
                return p
            prefix = '/api/samples/'
            if p.startswith(prefix):
                # project root = two levels up from this file (emseek/core/..)
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                return os.path.join(root, 'samples', p[len(prefix):])
            return p

        for f in files:
            try:
                # String form: path or base64/data URL
                if isinstance(f, str):
                    s = f.strip()
                    if not s:
                        continue
                    if s.startswith('data:image') or (s and s[0].isalnum() and len(s) > 64):
                        # looks like data URL or bare base64
                        path = save_base64_image(s, artifacts_dir, 'input')
                        image_paths.append(_as_abs(path))
                    else:
                        # Treat as a path; route by extension
                        s_mapped = _map_api_samples(s)
                        low = s_mapped.lower()
                        if low.endswith('.cif'):
                            cif_paths.append(_as_abs(s_mapped))
                        else:
                            image_paths.append(_as_abs(s_mapped))
                    continue

                # Dict form
                if isinstance(f, dict):
                    ftype = (f.get('type') or f.get('kind') or '').lower()
                    name = f.get('name') or ''
                    path = f.get('path')
                    if path:
                        path = _map_api_samples(path)
                        low = str(path).lower()
                        if low.endswith('.cif'):
                            cif_paths.append(_as_abs(path))
                        else:
                            image_paths.append(_as_abs(path))
                        continue
                    # CIF by content
                    if ftype == 'cif' or name.lower().endswith('.cif') or f.get('text'):
                        cif_name = name or f"structure_{int(time.time()*1000)}.cif"
                        cif_path = os.path.join(artifacts_dir, cif_name)
                        try:
                            with open(cif_path, 'w', encoding='utf-8') as out:
                                out.write(f.get('text', ''))
                        except Exception:
                            pass
                        cif_paths.append(_as_abs(cif_path))
                        continue
                    # Otherwise, treat as image-like data
                    data_b64 = f.get('base64') or f.get('data') or f.get('url')
                    if isinstance(data_b64, str):
                        img_path = save_base64_image(data_b64, artifacts_dir, 'input')
                        image_paths.append(_as_abs(img_path))
                        continue
            except Exception:
                # Skip malformed file entries silently
                continue

        # No global image registry; rely on unified payload propagation

        # Build unified agent payload and delegate to query()
        # Build unified payload with only present fields
        unified = {}
        if isinstance(text, str) and text.strip():
            unified["text"] = text.strip()
        if image_paths:
            unified["images"] = image_paths
        if cif_paths:
            unified["cifs"] = cif_paths
        if isinstance(bbox, dict) and bbox:
            unified["bbox"] = bbox
        if model:
            unified["model"] = model
        if options:
            unified["options"] = options

        for line in self.query(messages=unified, user_id=user_id):
            try:
                obj = json.loads(line)
                if 'chain_of_thought' in obj:
                    src = obj.get('agent') or ''
                    yield json.dumps(build_stream_step(src, text=obj.get('chain_of_thought', ''), images=obj.get('ref_images'))) + "\n"
                elif 'response' in obj:
                    yield json.dumps(build_stream_final(obj.get('response', ''), obj.get('user_id'), images=obj.get('ref_images'))) + "\n"
                else:
                    yield json.dumps(obj) + "\n"
            except Exception:
                # Not JSON? Pass-through raw line
                yield line

    # --- Agent-to-agent messaging envelope (non-breaking addition) --- #
    def send_message(self, sender, target_name, text=None, image=None):
        """
        Standardized agent-to-agent messaging. This does not alter existing
        agent behavior; it enables optional message passing with logging.
        """
        target = self.agents.get(target_name)
        info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from": getattr(sender, 'name', str(sender)),
            "to": target_name,
            "text": text,
            "image": image,
        }
        # Log for consistency
        self.record_history(sender.name if hasattr(sender, 'name') else 'Unknown', response=f"Send→{target_name}: {text}")
        if target and hasattr(target, 'receive_message'):
            target.receive_message(sender, text=text, image=image)

    def query(self, messages, user_id):
        """
        Run a Maestro-orchestrated multi-step query.
        In the streamed 'chain_of_thought':
        - On success (ok=True), return the task's 'message' (summary) and the 'text' (result).
        - On failure (ok=False), return error.message and error.fields.
        """
        self.record_history('User', messages)
        response_text, response_images = '', []

        for step in range(self.config.MAX_STEP):
            # 1) Ask Maestro what to do next
            task, agent_name, inputs, maestro_msg = self.agents['MaestroAgent'].forward(query=messages)

            # For UI/history, use Maestro's last high-level plan
            thought = getattr(self.agents['MaestroAgent'], 'last_thought', '') or ''
            self.record_history('MaestroAgent', (task, agent_name, inputs, maestro_msg), thought)

            # Stream Maestro plan as a step
            bot_text = f'{maestro_msg}\n'
            yield json.dumps({
                "chain_of_thought": bot_text,
                "ref_images": None,
                "agent": f"MaestroAgent -> {agent_name}" if 'None' not in agent_name else "MaestroAgent",
                "activity": self._agent_activity('MaestroAgent', 5),
            }) + "\n"

            # If Maestro says we're done
            if 'Final Answer' in task:
                response_text = maestro_msg
                break

            # 2) Route to the selected agent
            agent = self.agents[agent_name]
            # Merge inputs + maestro_msg into a single payload for the target agent
            payload = dict(inputs or {})
            if maestro_msg and ('task' not in payload):
                payload['task'] = maestro_msg

            out = agent.forward(payload if payload else maestro_msg)

            # 3) Normalize outputs (dict = unified; tuple = legacy)
            imgs = None
            history = None

            # Build the chain_of_thought content depending on success/failure
            if isinstance(out, dict):
                ok_flag = out.get('ok', None)
                imgs = out.get('images')
                history = out.get('meta')

                if ok_flag is True:
                    # SUCCESS: show message (summary) + text (detailed result)
                    msg = (out.get('message') or '').strip()
                    txt = (out.get('text') or '').strip()
                    # Join with a blank line only if both exist
                    joiner = "\n\n" if (msg and txt) else ""
                    text_for_stream = f"{msg}{joiner}{txt}".strip()

                elif ok_flag is False:
                    # FAILURE: show error.message + error.fields (if any)
                    err = out.get('error') or {}
                    err_msg = (err.get('message') or out.get('message') or 'Task failed').strip()
                    fields = err.get('fields') or []
                    if fields:
                        # keep short and explicit
                        fields_str = ", ".join(map(str, fields))
                        text_for_stream = f"{err_msg}\nMissing/invalid fields: {fields_str}"
                    else:
                        text_for_stream = err_msg

                else:
                    # No ok flag — fallback to previous behavior: message + text
                    text_for_stream = f"{out.get('message','')}{out.get('text','')}"

            else:
                # Legacy tuple: (text, image, history)
                text_legacy, image_legacy, history = out
                imgs = image_legacy
                text_for_stream = text_legacy

            # Stream the agent result
            imgs = imgs or None
            if isinstance(imgs, str) or (isinstance(imgs, list) and imgs):
                # Stream with images if present
                yield json.dumps({
                    "chain_of_thought": text_for_stream,
                    "ref_images": to_image_refs(imgs),
                    "agent": f"{agent_name} -> MaestroAgent",
                    "activity": self._agent_activity(agent_name, 5),
                }) + "\n"
            else:
                # Stream without images
                yield json.dumps({
                    "chain_of_thought": text_for_stream,
                    "ref_images": None,  # images are always converted via to_image_refs when present
                    "agent": f"{agent_name} -> MaestroAgent",
                    "activity": self._agent_activity(agent_name, 5),
                }) + "\n"

            # Persist step history
            self.record_history(agent_name, out, history)

        # Curate via GuardianAgent, then compose via ScribeAgent
        try:
            # Build question text to pass along (best effort)
            q_text = None
            if isinstance(messages, dict):
                q_text = messages.get('text') or messages.get('query')
            if not isinstance(q_text, str):
                q_text = str(messages)
            guardian_out = self.agents.get('GuardianAgent').forward({"text": q_text}) if 'GuardianAgent' in self.agents else None
            if isinstance(guardian_out, dict):
                # Stream GuardianAgent step: curation details
                try:
                    gmsg = (guardian_out.get('message') or '').strip()
                    gmeta = guardian_out.get('meta') or {}
                    agents_seen = gmeta.get('agents_seen') or []
                    sel = gmeta.get('selected_indices') or []
                    srcs = gmeta.get('sources') or []
                    parts = [gmsg] if gmsg else []
                    if sel:
                        parts.append(f"Selected items: {len(sel)}")
                    if agents_seen:
                        parts.append(f"Agents considered: {', '.join(map(str, agents_seen))[:200]}")
                    if srcs:
                        parts.append(f"Collected sources: {len(srcs)}")
                    guardian_step_text = ("\n\n".join(parts)).strip() or "Curating relevant history for final composition."
                    yield json.dumps({
                        "chain_of_thought": guardian_step_text,
                        "ref_images": to_image_refs(None),
                        "agent": "GuardianAgent -> ScribeAgent",
                        "activity": self._agent_activity('GuardianAgent', 5),
                    }) + "\n"
                except Exception:
                    pass
                # Record GuardianAgent history
                self.record_history('GuardianAgent', guardian_out, guardian_out.get('meta'))

                # If Guardian includes Scribe output, stream Scribe step and set final text
                scribe_out = (guardian_out.get('meta') or {}).get('scribe') if isinstance(guardian_out.get('meta'), dict) else None
                if isinstance(scribe_out, dict):
                    try:
                        msg = (scribe_out.get('message') or '').strip()
                        meta = scribe_out.get('meta') or {}
                        agents_seen = meta.get('agents_seen') or []
                        srcs = meta.get('sources') or []
                        parts = [msg] if msg else []
                        if agents_seen:
                            parts.append(f"Agents considered: {', '.join(map(str, agents_seen))[:200]}")
                        if srcs:
                            parts.append(f"Collected sources: {len(srcs)}")
                        scribe_step_text = ("\n\n".join(parts)).strip() or "Composing the final answer from recent history."
                        yield json.dumps({
                            "chain_of_thought": scribe_step_text,
                            "ref_images": to_image_refs(None),
                            "agent": "ScribeAgent -> Final",
                            "activity": self._agent_activity('ScribeAgent', 5),
                        }) + "\n"
                    except Exception:
                        pass
                    response_text = (scribe_out.get('text') or response_text or '').strip()
                    # Prefer images selected by ScribeAgent when provided
                    try:
                        imgs_scribe = scribe_out.get('images')
                        if imgs_scribe:
                            response_images = imgs_scribe
                    except Exception:
                        pass
                    # Record ScribeAgent history
                    self.record_history('ScribeAgent', scribe_out, scribe_out.get('meta'))
                else:
                    # Fallback to guardian text if scribe missing
                    gt = (guardian_out.get('text') or guardian_out.get('message') or '').strip()
                    if gt and not response_text:
                        response_text = gt
        except Exception:
            pass

        # Do not clear task_memory here; preserve full detailed records across turns

        # Use all images selected by ScribeAgent (if any). Do not print or expose paths directly.
        images_out = response_images if response_images else None

        final_response = {
            "response": response_text,
            "ref_images": to_image_refs(images_out),
            "user_id": user_id,
            "session_id": self.current_session_id,
        }
        yield json.dumps(final_response) + "\n"

    # --- Activity helpers --- #
    def _agent_activity(self, agent_name: str, k: int = 5) -> str:
        try:
            ag = self.agents.get(agent_name)
            if ag and hasattr(ag, 'recent_memory_text'):
                return ag.recent_memory_text(k)
        except Exception:
            pass
        return ""
