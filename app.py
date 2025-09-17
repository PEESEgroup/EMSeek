from flask import Flask, request, render_template, jsonify
from flask import Response
from flask import send_from_directory
import json
import os
import uuid
import cfg
from emseek.platform import Platform
from emseek.protocol import build_stream_final

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'examples'
# Explicitly allow large JSON bodies (actual gateway may still limit). Default 100 MB.
try:
    import os as _os
    _max_mb = int(_os.environ.get('MAX_UPLOAD_MB', '100'))
    app.config['MAX_CONTENT_LENGTH'] = _max_mb * 1024 * 1024
except Exception:
    pass

user_histories = dict()
emseek_platform = Platform(cfg)
emseek_platform.init_agent()

# Background warmup to reduce first-token latency (no functional change)
try:
    import threading
    def _background_warmup():
        try:
            # ScholarSeeker: prewarm PaperQA when enabled
            scholar = emseek_platform.agents.get('ScholarSeeker')
            if scholar and getattr(cfg, 'PREWARM_PAPERQA', False):
                try:
                    scholar.warmup(background=True)
                except Exception:
                    pass
        except Exception:
            pass
        # Avoid any Torch/CUDA pre-initialization here to stay fork-safe

    threading.Thread(target=_background_warmup, daemon=True, name="emseek-warmup").start()
except Exception:
    pass

def _client_ip() -> str:
    try:
        # Honor X-Forwarded-For when behind a proxy (take the first IP)
        xff = request.headers.get('X-Forwarded-For', '')
        if xff:
            return xff.split(',')[0].strip()
    except Exception:
        pass
    try:
        return request.remote_addr or 'unknown'
    except Exception:
        return 'unknown'


def set_userid(user_id):
    # Treat IP as the default user_id when not provided
    if not user_id:
        user_id = _client_ip()
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_history_folder = os.path.join('./history', user_id)
    emseek_platform.setup_working_folder(user_history_folder)

    return user_id


# ---------------- Session storage helpers ---------------- #
import time

SESSIONS_DIR_NAME = 'sessions'


def _user_root(user_id: str) -> str:
    return os.path.join('./history', user_id)


def _sessions_dir(user_id: str) -> str:
    return os.path.join(_user_root(user_id), SESSIONS_DIR_NAME)


def _sessions_index_path(user_id: str) -> str:
    return os.path.join(_sessions_dir(user_id), 'index.json')


def _ensure_dirs(user_id: str):
    os.makedirs(_sessions_dir(user_id), exist_ok=True)


def _read_json(path: str, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    try:
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    except Exception:
        return str(time.time())


# Active session id per user (in-memory convenience)
active_session_for_user = {}


def _new_session(user_id: str) -> str:
    _ensure_dirs(user_id)
    sid = str(uuid.uuid4())
    sess = {
        'id': sid,
        'title': 'Untitled',
        'created_at': _now_iso(),
        'updated_at': _now_iso(),
        'messages': []  # [{role: 'user'|'assistant', content: str, images: [ref...]}]
    }
    # persist session file
    path = os.path.join(_sessions_dir(user_id), f'{sid}.json')
    _write_json(path, sess)
    # update index (prepend)
    idx_path = _sessions_index_path(user_id)
    idx = _read_json(idx_path, [])
    idx = [i for i in idx if i.get('id') != sid]
    idx.insert(0, {'id': sid, 'title': sess['title'], 'created_at': sess['created_at'], 'updated_at': sess['updated_at']})
    # keep only 3; optionally prune files
    if len(idx) > 3:
        for drop in idx[3:]:
            try:
                drop_path = os.path.join(_sessions_dir(user_id), f"{drop['id']}.json")
                if os.path.exists(drop_path):
                    os.remove(drop_path)
            except Exception:
                pass
        idx = idx[:3]
    _write_json(idx_path, idx)
    active_session_for_user[user_id] = sid
    return sid


def _load_session(user_id: str, sid: str):
    path = os.path.join(_sessions_dir(user_id), f'{sid}.json')
    return _read_json(path, None)


def _save_session(user_id: str, sess: dict):
    sid = sess.get('id')
    if not sid:
        return
    sess['updated_at'] = _now_iso()
    path = os.path.join(_sessions_dir(user_id), f'{sid}.json')
    _write_json(path, sess)
    # update index title and timestamps
    idx_path = _sessions_index_path(user_id)
    idx = _read_json(idx_path, [])
    found = False
    for it in idx:
        if it.get('id') == sid:
            it['title'] = sess.get('title', it.get('title', 'Untitled'))
            it['updated_at'] = sess.get('updated_at', it.get('updated_at', _now_iso()))
            found = True
            break
    if not found:
        idx.insert(0, {'id': sid, 'title': sess.get('title', 'Untitled'), 'created_at': sess.get('created_at', _now_iso()), 'updated_at': sess.get('updated_at', _now_iso())})
    # keep only 3 (do not prune here to avoid accidental deletion of unrelated sessions)
    if len(idx) > 3:
        idx = idx[:3]
    _write_json(idx_path, idx)


def _summarize_title_with_llm(conversation_text: str) -> str:
    try:
        maestro = emseek_platform.agents.get('MaestroAgent')
        if not maestro:
            return ''
        prompt = (
            "You are a helpful assistant. Read the following brief chat excerpt and produce a concise, descriptive English title summarizing the task. "
            "Constraints: 4-8 words, Title Case, no punctuation except spaces, no quotes.\n\n"
            f"Chat:\n{conversation_text[:2000]}\n\nTitle:"
        )
        out = maestro.llm_call(messages=[{"role": "system", "content": prompt}], max_tokens=16, temperature=0.2)
        if not isinstance(out, str):
            return ''
        title = out.strip().splitlines()[0].strip()
        # basic cleanup
        title = title.strip('"').strip("'")
        if len(title) > 64:
            title = title[:64]
        return title or ''
    except Exception:
        return ''

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/samples/<path:filename>')
@app.route('/api/samples/<path:filename>')
def serve_sample(filename):
    """Serve example EM images from the root-level samples/ directory for the frontend demos.

    Expose under both `/samples/...` and `/api/samples/...` so deployments
    with nginx that only proxy `/api` still load images.
    """
    samples_dir = os.path.join(app.root_path, 'samples')
    return send_from_directory(samples_dir, filename)

# Disable buffering on NDJSON streaming for faster client updates
@app.after_request
def _add_streaming_headers(resp: Response):
    try:
        if resp.mimetype == 'application/x-ndjson':
            resp.headers['Cache-Control'] = 'no-cache, no-transform'
            resp.headers['X-Accel-Buffering'] = 'no'
            # ensure charset to avoid proxies touching payload
            resp.headers['Content-Type'] = 'application/x-ndjson; charset=utf-8'
    except Exception:
        pass
    return resp

# Single unified API route
@app.route('/api', methods=['POST'])
def api_unified():
    """
    Single JSON Dict API for all interactions.
    Accepts fields like:
      - user_id: optional string; generated if not provided
      - action: optional string, e.g., 'cleanup' to reset state
      - text: user query
      - files: list of image/CIF items. Each item can be a base64 data URL string, a path string, or a dict like {type: 'image'|'cif', data|text, name?, path?}
      - bbox: rect dict
      - model: independent model selection (e.g., 'general_model')
      - options: extra options (elements, top_k, ...)
      - agent: routing hint

    Always streams NDJSON frames (type: 'step'|'final').
    """
    data = request.get_json() or {}
    # Prefer provided id; otherwise treat client IP as account id
    user_id = set_userid(data.get('user_id') or _client_ip())
    data = dict(data)
    data['user_id'] = user_id

    # Control operation: cleanup/reset
    action = (data.get('action') or '').strip().lower()
    if action == 'cleanup' or action == 'clear':
        try:
            emseek_platform.clear()
        except Exception:
            pass
        # Also drop stored history mapping for this user
        try:
            if user_id in user_histories:
                del user_histories[user_id]
        except Exception:
            pass
        # Clear active session mapping; do not pre-create a new one
        try:
            active_session_for_user.pop(user_id, None)
        except Exception:
            pass
        # Reset platform context to user root (no session)
        try:
            emseek_platform.set_session(user_id=user_id, session_id=None)
        except Exception:
            pass

        def _final_only():
            yield json.dumps(build_stream_final("History has been cleaned up.", user_id)) + "\n"

        return Response(_final_only(), mimetype='application/x-ndjson')

    # Determine session to use
    session_id = data.get('session_id') or active_session_for_user.get(user_id)
    if not session_id:
        session_id = _new_session(user_id)
    else:
        # Ensure it exists; otherwise create
        sess = _load_session(user_id, session_id)
        if not isinstance(sess, dict):
            session_id = _new_session(user_id)
    active_session_for_user[user_id] = session_id
    # Inform platform so it can isolate artifacts/logs and memory
    try:
        emseek_platform.set_session(user_id=user_id, session_id=session_id)
    except Exception:
        pass

    # Merge prior session context into query text to maintain continuity
    try:
        # Minimal lightweight context: last assistant message as background
        sess = _load_session(user_id, session_id)
        prior = []
        if sess and isinstance(sess.get('messages'), list):
            for m in sess['messages'][-4:]:  # last few turns
                role = m.get('role')
                content = (m.get('content') or '').strip()
                if role and content:
                    prior.append(f"{role}: {content}")
        if prior:
            user_text = (data.get('text') or '').strip()
            merged = ("Context from earlier conversation (for reference, do not repeat verbatim):\n" + "\n".join(prior) + "\n\nUser question: " + user_text).strip()
            data['text'] = merged
    except Exception:
        pass

    # Capture the user message into the session before calling platform
    try:
        sess = _load_session(user_id, session_id)
        if not sess:
            session_id = _new_session(user_id)
            sess = _load_session(user_id, session_id)
        # Store raw user text separately (without embedded context) for display
        incoming = (request.get_json() or {})
        display_text = incoming.get('text')
        # Try to capture any user-attached images for later restoration
        user_images = []
        try:
            files_in = incoming.get('files')
            if isinstance(files_in, list):
                for f in files_in:
                    if isinstance(f, dict) and (f.get('type') or f.get('kind')) == 'image':
                        img_data = f.get('data') or f.get('base64') or f.get('url') or f.get('path')
                        if isinstance(img_data, str) and img_data:
                            # Normalize to refs (prefer base64 if provided)
                            from emseek.core.protocol import to_image_refs as _tir
                            user_images.extend(_tir(img_data))
        except Exception:
            pass
        if display_text or user_images:
            msg = {'role': 'user', 'content': display_text or ''}
            if user_images:
                msg['images'] = user_images
            sess['messages'].append(msg)
            _save_session(user_id, sess)
    except Exception:
        pass

    # Default path: unified multimodal query handling (wrap to capture stream)
    def _wrapped_stream():
        import queue as _queue
        import threading as _threading

        # Configurable heartbeat interval (seconds). 0 disables periodic heartbeats.
        try:
            hb_interval = int(getattr(cfg, 'STREAM_HEARTBEAT_INTERVAL_SEC', 10))
        except Exception:
            hb_interval = 10

        # Send a quick heartbeat frame immediately to open the stream.
        try:
            yield json.dumps({
                "type": "heartbeat",
                "time": _now_iso(),
                "user_id": user_id
            }) + "\n"
        except Exception:
            pass

        # Prepare a queue to interleave backend chunks with periodic heartbeats.
        q: _queue.Queue = _queue.Queue(maxsize=256)
        _SENTINEL = object()

        def _worker():
            try:
                for chunk in emseek_platform.query_unified(payload=data):
                    try:
                        q.put(chunk, timeout=5)
                    except Exception:
                        # If client/backpressure blocks, drop chunk to avoid deadlock
                        pass
            except Exception as e:
                try:
                    err_msg = f"Server error: {e}"
                    q.put(json.dumps(build_stream_final(err_msg, user_id)) + "\n")
                except Exception:
                    pass
            finally:
                try:
                    q.put(_SENTINEL)
                except Exception:
                    pass

        # Emit an initial session-binding frame so the client can pin to this session
        try:
            yield json.dumps({
                "type": "session",
                "time": _now_iso(),
                "user_id": user_id,
                "session_id": session_id,
            }) + "\n"
        except Exception:
            pass

        t = _threading.Thread(target=_worker, name="api-stream-worker", daemon=True)
        t.start()

        final_text = None
        images_refs = None
        steps_collected = []

        # Drain queue with periodic heartbeat while the worker runs
        while True:
            try:
                item = q.get(timeout=hb_interval if hb_interval > 0 else None)
            except Exception:
                # Timeout: emit heartbeat to keep the connection alive
                try:
                    yield json.dumps({
                        "type": "heartbeat",
                        "time": _now_iso(),
                        "user_id": user_id
                    }) + "\n"
                except Exception:
                    pass
                continue

            if item is _SENTINEL:
                break

            # Forward backend chunk to client
            try:
                yield item
            except Exception:
                # If yielding fails (client closed), stop early
                break

            # Capture assistant outputs for history
            try:
                obj = json.loads(item)
                if obj.get('type') == 'step':
                    steps_collected.append({
                        'time': obj.get('time'),
                        'source': obj.get('source'),
                        'text': obj.get('text'),
                        'images': obj.get('images'),
                    })
                elif obj.get('type') == 'final':
                    final_text = obj.get('text') or obj.get('response')
                    images_refs = obj.get('images') or obj.get('ref_images')
            except Exception:
                pass

        # Persist assistant final into session and derive title
        try:
            sess = _load_session(user_id, session_id)
            if sess and final_text:
                turn = {'role': 'assistant', 'content': final_text}
                if images_refs:
                    turn['images'] = images_refs
                if steps_collected:
                    turn['steps'] = steps_collected
                sess['messages'].append(turn)
                # Generate/update title if it remains Untitled or short
                if not sess.get('title') or sess.get('title') == 'Untitled' or len(sess.get('title', '')) < 4:
                    # Build a small transcript for LLM titling
                    parts = []
                    for m in sess['messages'][:2]:
                        parts.append(f"{m.get('role')}: {m.get('content','')}")
                    title = _summarize_title_with_llm("\n".join(parts))
                    if title:
                        sess['title'] = title
                _save_session(user_id, sess)
        except Exception:
            pass

    return Response(_wrapped_stream(), mimetype='application/x-ndjson')


# ---------------- History API ---------------- #
@app.route('/api/history/list', methods=['GET'])
def api_history_list():
    user_id = set_userid(request.args.get('user_id') or _client_ip())
    idx = _read_json(_sessions_index_path(user_id), [])
    # Ensure at most 3 entries
    idx = idx[:3]
    return jsonify({'user_id': user_id, 'sessions': idx})


@app.route('/api/history/session/<sid>', methods=['GET'])
def api_history_session(sid):
    user_id = set_userid(request.args.get('user_id') or _client_ip())
    sess = _load_session(user_id, sid)
    if not sess:
        return jsonify({'ok': False, 'error': 'Session not found', 'user_id': user_id}), 404
    # Mark active
    active_session_for_user[user_id] = sid
    return jsonify({'ok': True, 'user_id': user_id, 'session': sess})


@app.route('/api/history/session/delete', methods=['POST'])
def api_history_session_delete():
    """Delete an entire chat session for the user.

    Request JSON:
      - user_id: optional string; defaults to client IP
      - session_id: required string (or 'id'/'sid')

    Returns JSON { ok: bool, user_id, deleted_session_id, sessions }
    """
    try:
        data = request.get_json() or {}
        user_id = set_userid(data.get('user_id') or _client_ip())
        sid = data.get('session_id') or data.get('id') or data.get('sid')
        if not sid:
            return jsonify({'ok': False, 'error': 'Missing session_id', 'user_id': user_id}), 400

        # Delete session file
        try:
            path = os.path.join(_sessions_dir(user_id), f'{sid}.json')
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

        # Update index
        idx_path = _sessions_index_path(user_id)
        idx = _read_json(idx_path, [])
        idx = [i for i in idx if i.get('id') != sid]
        _write_json(idx_path, idx)

        # Clear active mapping if it pointed to this session
        try:
            if active_session_for_user.get(user_id) == sid:
                active_session_for_user.pop(user_id, None)
                # Reset platform to user root (no active session)
                try:
                    emseek_platform.set_session(user_id=user_id, session_id=None)
                except Exception:
                    pass
        except Exception:
            pass

        return jsonify({'ok': True, 'user_id': user_id, 'deleted_session_id': sid, 'sessions': idx})
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Unexpected error: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port='8000')
