# EMSeek Runtime Architecture

## High-level layout
- `app.py` (`app.py:1`) exposes a Flask service that backs both the browser UI and programmatic clients. It mounts a single `/api` JSON endpoint that streams NDJSON frames to the caller and a small set of helper routes (samples, history CRUD).
- `cfg.py` (`cfg.py:1`) centralises environment-driven settings such as model choices, file-system paths, concurrency limits, and feature toggles. Every agent receives the same configuration object via the `Platform` so they can read shared knobs.
- `emseek/core/platform.py` wraps the multi-agent runtime. It owns the agent registry, cross-agent memory, working directories, and all logging/artifact plumbing.
- Agents live under `emseek/agents/*`. Each subclass of `Agent` focuses on a specialised capability (vision segmentation, literature review, analysis tooling, structure retrieval, energy prediction, curation, and final drafting).

## Startup sequence
1. Flask initialises in `app.py` and constructs a single global `Platform` instance, passing it the imported `cfg` module.
2. `Platform.__init__` (`emseek/core/platform.py:22`) chooses or creates a working directory, sets up log/artefact folders, and prepares rolling memories (`long_memory`, `task_memory`, and pseudo-user history).
3. `Platform.init_agent()` (`emseek/core/platform.py:98`) instantiates every built-in agent and stores them in `self.agents` for later routing. This happens once at process start so the same objects handle all requests.
4. An optional warm-up thread (`app.py:24`) can pre-invoke long-loading models such as PaperQA when the relevant feature flag is enabled.

## Request lifecycle
1. Clients POST to `/api` with a JSON payload that may include text, file references/base64 blobs, optional bounding boxes, agent hints, and model overrides. `app.py` normalises caller identity, persists session scaffolding, and merges a thin slice of prior conversation into the current question for continuity.
2. Before invoking the multi-agent core, the raw message is appended to the per-user session log (`history/<user>/sessions/<id>.json`) so conversational state survives restarts.
3. `Platform.query_unified()` (`emseek/core/platform.py:201`) receives the payload, writes any base64 images to session-specific artefact folders, flattens the multimodal inputs, and then delegates to `Platform.query()` for orchestration. It yields NDJSON frames on the fly so clients can render progressive updates.
4. The Flask route wraps the generator with a heartbeat mechanism to keep proxies and browsers from timing out. It also intercepts the final frame to persist the assistant response back into session storage and optionally title the chat via the Maestro agent.

## Multi-agent orchestration
- `Platform.query()` executes up to `cfg.MAX_STEP` reasoning loops. Each turn calls `MaestroAgent.forward()` to decide the next action. Maestro returns either a `|Final Answer|` directive or a `|Action|<agent>|<inputs>|<message>` tuple.
- If Maestro selects an agent, the chosen controller receives a merged payload that contains both the structured inputs and Maestro’s natural-language instruction. Every agent returns a unified dict describing success/failure, textual output, image artefacts, and rich metadata.
- Each agent result triggers a streaming `step` frame so clients can visualise the chain-of-thought. The sanitised response is also recorded in both agent-specific memory and global `task_memory` for downstream agents (like Guardian/Scribe) to reference.
- When Maestro emits a final directive (or the loop exhausts), `GuardianAgent` curates the accumulated history and forwards a condensed context to `ScribeAgent`. Scribe composes the final answer and selects any supporting images. The combined result is streamed as the `final` frame and returned to the caller.

## Memory, logging, and artefacts
- The working directory tree is rooted at `history/<user>/[sessions/<session>]`. Each session has dedicated `logs/` and `artifacts/` subfolders so intermediate outputs never clash across users.
- `Platform.record_history()` converts inline base64 image payloads to files for auditability, persists append-only JSONL logs per agent, and updates the in-memory `task_memory` list used for context injection.
- Session metadata (`index.json` plus individual conversation transcripts) lives under the same tree and powers the history API routes exposed in `app.py`.

## Agent ecosystem and boundaries
- `MaestroAgent` routes tasks, calling other agents and occasionally invoking a multimodal model purely to describe inputs before routing decisions.
- `SegMentorAgent` handles Unet++-based segmentation and produces textual descriptions of overlays through an MLLM prompt.
- `AnalyzerHubAgent` validates natural-language analysis requests and maps them onto concrete python tools in `emseek.tools`, including automatic parameter repair.
- `ScholarSeekerAgent` orchestrates PaperQA and CORE searches with citation management for literature questions.
- `CrystalForgeAgent` retrieves CIF structures that match a query image and element hints, scores candidates, and renders quick-look galleries.
- `MatProphetAgent` runs four energy-prediction models plus a mixture-of-experts combiner to report per-atom and total energies.
- `GuardianAgent` curates cross-agent history snippets into a context bundle.
- `ScribeAgent` writes the final user-facing answer, preserving numeric and visual evidence.

## Configuration and extensibility
- Environment variables (read in `cfg.py`) control model identifiers, concurrency ceilings, data-library roots, and optional behaviours such as PaperQA warm-up or streaming heartbeat cadence. Agents read these knobs via their shared `config` attribute.
- New agents or tools can be registered by extending `Platform.init_agent()` or adding entries to `emseek.tools`. Because Maestro emits structured directives, additional controllers must implement the same unified input/output shape to integrate smoothly.
