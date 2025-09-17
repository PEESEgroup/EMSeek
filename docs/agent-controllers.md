# Agent LLM Controllers

All EMSeek agents share the `Agent` base class, which provides helper facilities: optional guard-LLM input normalisation, unified payload flattening, per-agent memory, and helpers to record artefacts. The following sections describe how each specialised agent uses LLM or MLLM controllers to validate inputs, plan actions, and generate outputs.

## MaestroAgent (`emseek/agents/maestro_agent.py`)
- Normalises incoming queries, converts image references to session-local paths, and keeps a running history of task memory for context.
- Optionally calls `_parse_images_to_text` to obtain short multimodal bullet points that help the text-only routing LLM understand visual inputs.
- Builds a contract-first prompt listing available agents, their recent state, any parsed image notes, and the current user query. The LLM must return either a `|Final Answer|None|...|` directive or an `|Action|<agent>|<inputs>|<message>` directive.
- Parses the directive, sanitises JSON inputs for the target agent, and stores the parsed “thought” for streaming back to clients.

## SegMentorAgent (`emseek/agents/segmentor_agent.py`)
- `controller_validate_and_fix` fuses inputs from `images`, `image_path`, and recent memory. When no file is provided, it asks an LLM to recover a usable path from the agent’s own history. Bounding boxes are optional and gracefully discarded if malformed.
- Runs the loaded Unet++ checkpoint or an Otsu fallback to produce masks and overlays. Artefacts are stored in the session’s `artifacts/` directory.
- Calls the configured MLLM with a domain-specific prompt to describe salient morphology, imaging conditions, and overlay interpretation. The output becomes the human-readable report included in the unified response.

## AnalyzerHubAgent (`emseek/agents/analyzerhub_agent.py`)
- `controller_validate_and_fix` enforces that either `tool` or `task` exists. If missing, it consults an LLM to infer the task from recent memory and backfills parameters such as `image_path` and `cif_path` using the unified payload.
- Performs lightweight validation for each tool, auto-selecting defaults and ensuring referenced files exist. Missing requirements surface as actionable `{ok: False, error.fields}` responses.
- If the caller omits a concrete tool, `_decide_tool_llm` asks the LLM to pick the best match from the current tool registry, with heuristics as fallback.
- After `run_tool` executes, the agent optionally captions generated images via an MLLM and asks the LLM for a short success sentence for Maestro. The unified response includes a status message, detailed text, image paths, and rich meta diagnostics.

## ScholarSeekerAgent (`emseek/agents/scholarseeker_agent.py`)
- Validates that a research question is present. When absent, the controller feeds recent memory into an LLM that must return `{ "query": "..." }`.
- Manages asynchronous PaperQA document caches, optional CORE searches, and citation tracking. Retrieved PDFs are stored under the configured `PDF_FOLDER`.
- After synthesising an answer, it asks a small LLM for a concise recap mentioning citation counts and answer length, which Maestro surfaces in the reasoning trace.

## CrystalForgeAgent (`emseek/agents/crystalforge_agent.py`)
- `controller_validate_and_fix` ensures a usable EM image exists, normalises element lists, and can infer elements from filenames, recent memory, or an LLM extraction step.
- Once validation passes, it runs segmentation (CUDA Unet++ or Otsu), filters the CIF index by elements, scores candidates, and renders gallery tiles with PIL.
- Uses a short LLM prompt to craft the one-line status message highlighting element coverage and top-scoring CIF.

## MatProphetAgent (`emseek/agents/matprophet_agent.py`)
- The controller reconciles `cif_path` across raw payloads, unified inputs, and history-derived hints. An LLM helper can recover the most recent valid path when the user omits it.
- After loading the structure, it evaluates MatterSim, UMA, ORBv3, and MACE predictors, then blends per-atom energies via a mixture-of-experts checkpoint (if available) or a simple mean fallback.
- Compiles a comprehensive JSON summary (energies, forces, stress proxies, uncertainty, structural metadata) and sends it to an LLM that returns bullet-point prose, feeding the response text.

## GuardianAgent (`emseek/agents/guardian_agent.py`)
- Gathers the last ~80 entries from `task_memory`, preserving message text, agent names, image summaries, and compact metadata.
- Asks an LLM to select the indices most relevant to the current question; if the LLM abstains, a heuristic token-overlap fallback runs.
- The curated snippets, along with collected sources and agent provenance, become the context delivered to Scribe. Guardian’s response also records which agents were consulted and any embedded Scribe output.

## ScribeAgent (`emseek/agents/scribe_agent.py`)
- Collects a detailed history transcript (without aggressive truncation), recent sources, and agent participation lists.
- Issues a strict system instruction before passing the user question and history to the LLM, ensuring the final answer maintains evidence, uses neutral tone, and avoids proposing future plans.
- Post-processes the LLM text: masks file paths, extracts inline images (converting to base64 refs), and augments with additional artefacts from history until reaching the configured `FINAL_IMAGES_TOPK` limit.

## Maestro-adjacent helpers in Platform (`emseek/core/platform.py`)
- `Platform.query_unified` converts streamed base64 payloads to session-scoped files so controllers handle only local paths.
- `Platform.query` streams Maestro thoughts, agent outputs, Guardian curation, and Scribe composition back to the caller while logging each step into `task_memory`. This shared memory is the backbone that powers every history-aware controller described above.
