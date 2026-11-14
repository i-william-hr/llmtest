#!/usr/bin/env python3
"""llmtest - run one prompt against several models at once, side by side.

Providers: OpenRouter, Meta, Groq, Cerebras and NVIDIA (OpenAI-compatible
chat/completions),
plus Google via its native attributed-QA endpoint.
Streaming: the browser opens one SSE stream per selected model; the Flask
worker blocks on the provider response and relays tokens (or, for providers
that don't stream usefully, the complete answer as one event), so nginx only
ever sees a single response per request.

Run:  ./run.sh   ->  http://127.0.0.1:[REDACTED:loopback-port]/llmtest/  (nginx: https://[REDACTED:internal-hostname]/)
Auth: none of its own. nginx puts it behind the SSO gate; see
      [REDACTED:server-path] (proxy-llmtest.conf was
      retired when the vhost moved to sites/).
"""

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager

import requests
from flask import (Flask, Response, jsonify, render_template, request)
from werkzeug.middleware.proxy_fix import ProxyFix

# ----------------------------------------------------------------- config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Minimal .env loader, so settings are picked up however the app is
    launched (systemd EnvironmentFile, run.sh, or directly).

    Runs before the settings below are read, so every one of them can come
    from .env.
    """
    path = path or os.path.join(BASE_DIR, ".env")
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

HOST = os.environ.get("LLMTEST_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLMTEST_PORT", "[REDACTED:loopback-port]"))
PREFIX = os.environ.get("LLMTEST_PREFIX", "/llmtest")
DB_PATH = os.environ.get("LLMTEST_DB", os.path.join(BASE_DIR, "llmtest.db"))

# Provider credentials live in .env, never in this file. They default to
# empty: a missing key should fail visibly on use, not silently authenticate
# as nobody against a real provider.
#
# All three are treated as COMPROMISED -- they were inline here, and this file
# is readable by the service account and by every backup of the tree. Rotate
# each in its own dashboard, then update .env and restart.
PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter",
        "base_url": os.environ.get("LLMTEST_OPENROUTER_BASE_URL",
                                   "https://openrouter.ai/api/v1"),
        "api_key": os.environ.get("LLMTEST_OPENROUTER_API_KEY", ""),
        "extra_headers": {
            "HTTP-Referer": "https://[REDACTED:internal-hostname]/",
            "X-Title": "llmtest",
        },
    },
    "meta": {
        "label": "Meta",
        "base_url": os.environ.get("LLMTEST_META_BASE_URL",
                                   "https://api.meta.ai/v1"),
        "api_key": os.environ.get("LLMTEST_META_API_KEY", ""),
        "extra_headers": {},
        # /models lists non-text models too (sam needs images, transcribe
        # needs audio, image is image-gen); only spark-* speaks plain chat
        "chat_models": lambda mid: mid.startswith("muse-spark-"),
        # streaming returns empty deltas only, so fetch whole and relay once
        "stream": False,
    },
    "google": {
        "label": "Google",
        "base_url": os.environ.get("LLMTEST_GOOGLE_BASE_URL",
                                   "https://generativelanguage.googleapis.com/v1beta"),
        "api_key": os.environ.get("LLMTEST_GOOGLE_API_KEY", ""),
        "extra_headers": {},
        "key_style": "query",       # ?key= instead of Authorization header
        "kind": "aqa",              # native generateAnswer, not chat/completions
        "chat_models": lambda mid: mid == "aqa",
        "stream": False,
    },
    "groq": {
        "label": "Groq",
        "base_url": os.environ.get("LLMTEST_GROQ_BASE_URL",
                                   "https://api.groq.com/openai/v1"),
        "api_key": os.environ.get("LLMTEST_GROQ_API_KEY", ""),
        "extra_headers": {},
        # /models also lists transcription models (whisper-*); only the
        # chat-completion ones belong in the picker
        "chat_models": lambda mid: not mid.startswith("whisper-"),
    },
    "cerebras": {
        "label": "Cerebras",
        "base_url": os.environ.get("LLMTEST_CEREBRAS_BASE_URL",
                                   "https://api.cerebras.ai/v1"),
        "api_key": os.environ.get("LLMTEST_CEREBRAS_API_KEY", ""),
        "extra_headers": {},
        # /models also lists audio models (whisper-*); only the
        # chat-completion ones belong in the picker
        "chat_models": lambda mid: not mid.startswith("whisper-"),
    },
    "nvidia": {
        "label": "NVIDIA",
        "base_url": os.environ.get("LLMTEST_NVIDIA_BASE_URL",
                                   "https://integrate.api.nvidia.com/v1"),
        "api_key": os.environ.get("LLMTEST_NVIDIA_API_KEY", ""),
        "extra_headers": {},
    },
}

MODEL_REFRESH = int(os.environ.get("LLMTEST_MODEL_REFRESH", "300"))
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "llmtest/1.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llmtest")


def scrub(text):
    """Strip provider API keys out of a string before it is shown or logged.

    The Google provider authenticates with ?key=, so the credential ends up in
    the request URL -- and requests/urllib3 copy the full URL into the text of
    raise_for_status() and MaxRetryError. Without this, any 4xx or 5xx from
    upstream publishes the key in the API response and in the journal.
    """
    out = str(text)
    for pconf in PROVIDERS.values():
        key = pconf.get("api_key") or ""
        if key:
            out = out.replace(key, "<redacted>")
    return out

# ------------------------------------------------------------------ model ---
class ModelRegistry:
    """Remote model lists, cached in sqlite, refreshed in the background."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_refresh = 0.0

    def fetch_remote(self):
        items = []
        errors = {}
        for pkey, pconf in PROVIDERS.items():
            try:
                if pconf.get("key_style") == "query":
                    r = HTTP.get(pconf["base_url"] + "/models",
                                 params={"key": pconf["api_key"]}, timeout=30)
                else:
                    r = HTTP.get(pconf["base_url"] + "/models",
                                 headers=self._headers(pkey), timeout=30)
                r.raise_for_status()
                for m in r.json().get("models" if pconf.get("key_style") == "query" else "data", []):
                    mid = m.get("name", "").split("/")[-1] if pconf.get("key_style") == "query" else m.get("id")
                    if not mid:
                        continue
                    label = (m.get("displayName") or mid) \
                        if pconf.get("key_style") == "query" else (m.get("name") or mid)
                    chat_filter = pconf.get("chat_models")
                    if chat_filter is not None:
                        if not chat_filter(mid):
                            continue
                        chat_ok = True
                    else:
                        # providers without an explicit filter are assumed
                        # chat-capable when they say type=chat (or say nothing)
                        chat_ok = m.get("type", "chat") == "chat"
                    pricing = m.get("pricing") or {}
                    try:
                        free = float(pricing.get("prompt", 1)) == 0.0 and \
                               float(pricing.get("completion", 1)) == 0.0
                    except (TypeError, ValueError):
                        free = False
                    items.append({
                        "provider": pkey,
                        "id": mid,
                        "label": label,
                        "chat_ok": chat_ok,
                        "free": bool(m.get("is_free")) or free,
                        # openrouter labels look like "OpenAI: GPT-4o-mini"
                        "group": m.get("group") or \
                                 (label.split(":")[0].strip() if ":" in label else "other"),
                    })
            except Exception as exc:  # keep stale cache on failure
                errors[pkey] = f"{type(exc).__name__}: {scrub(exc)}"
                log.warning("model list fetch failed for %s: %s: %s",
                            pkey, type(exc).__name__, scrub(exc))
        return items, errors

    @staticmethod
    def _headers(pkey):
        pconf = PROVIDERS[pkey]
        h = {"Authorization": "Bearer " + pconf["api_key"]}
        h.update(pconf["extra_headers"])
        return h

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.last_refresh < MODEL_REFRESH:
                return
            items, errors = self.fetch_remote()
            if items:
                with db() as con:
                    now = int(time.time())
                    con.executemany(
                        "INSERT OR REPLACE INTO models(provider,model_id,label,"
                        " chat_ok,free,grp,updated_at) VALUES(?,?,?,?,?,?,?)",
                        [(it["provider"], it["id"], it["label"],
                          int(it["chat_ok"]), int(it["free"]), it["group"], now)
                         for it in items])
                log.info("refreshed model lists (%d models)", len(items))
            self.last_refresh = time.time()
            for pkey, msg in errors.items():
                with db() as con:
                    con.execute(
                        "UPDATE providers SET last_error=?, updated_at=? WHERE key=?",
                        (msg, int(time.time()), pkey))

    def all_models(self):
        with db() as con:
            rows = con.execute(
                "SELECT provider, model_id, label, free, grp FROM models "
                "WHERE chat_ok=1 ORDER BY provider, label").fetchall()
        out = {}
        for r in rows:
            out.setdefault(r["provider"], []).append(
                {"id": r["model_id"], "label": r["label"],
                 "free": bool(r["free"]), "group": r["grp"]})
        return out


# ----------------------------------------------------------------- flask ---
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
# No app.secret_key and no session: the app keeps no state between requests,
# so there is nothing for one to sign. The gate in front of it is nginx's.


class PrefixMiddleware:
    """Serves the app under PREFIX (nginx passes the full path through)."""

    def __init__(self, wsgi, prefix):
        self.wsgi = wsgi
        self.prefix = prefix.rstrip("/")

    def __call__(self, environ, start_response):
        # Deliberately does NOT honor client-supplied HTTP_X_SCRIPT_NAME: a
        # caller controls neither SCRIPT_NAME handling nor the 302 target.
        # The prefix is deployment configuration, not request data.
        script = self.prefix
        path = environ.get("PATH_INFO", "")
        if path == script or path.startswith(script + "/"):
            environ["SCRIPT_NAME"] = script
            environ["PATH_INFO"] = path[len(script):] or "/"
            return self.wsgi(environ, start_response)
        if path in ("", "/"):
            start_response("302 Found",
                           [("Location", script + "/"),
                            ("Content-Type", "text/plain; charset=utf-8")])
            return [b"redirecting...\n"]
        start_response("404 Not Found",
                       [("Content-Type", "text/plain; charset=utf-8")])
        return [b"not found (this app lives under " + script.encode() + b")\n"]


app.wsgi_app = PrefixMiddleware(
    ProxyFix(app.wsgi_app, x_proto=1, x_host=1), PREFIX)

registry = ModelRegistry()


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS models(
            provider   TEXT NOT NULL,
            model_id   TEXT NOT NULL,
            label      TEXT,
            chat_ok    INTEGER DEFAULT 1,
            free       INTEGER DEFAULT 0,
            grp        TEXT DEFAULT '',
            updated_at INTEGER,
            PRIMARY KEY(provider, model_id)
        );
        CREATE TABLE IF NOT EXISTS providers(
            key        TEXT PRIMARY KEY,
            last_error TEXT,
            updated_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS chats(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL DEFAULT 'new chat',
            models     TEXT NOT NULL DEFAULT '[]',
            system     TEXT NOT NULL DEFAULT '',
            temperature REAL,
            max_tokens INTEGER,
            created_at INTEGER,
            updated_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS messages(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role       TEXT NOT NULL,             -- user | assistant
            provider   TEXT DEFAULT '',
            model_id   TEXT DEFAULT '',
            text       TEXT NOT NULL DEFAULT '',
            reasoning  TEXT NOT NULL DEFAULT '',
            usage      TEXT DEFAULT '',
            error      TEXT DEFAULT '',
            created_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
        """)
        for pkey in PROVIDERS:
            con.execute("INSERT OR IGNORE INTO providers(key) VALUES(?)", (pkey,))


# ------------------------------------------------------------- access ---
# There is deliberately no auth code in this file. nginx gates every path
# under /llmtest/ (including /llmtest/static/ and the SSE streams) with an
# auth_request subrequest against the SSO app, so anything that reaches the
# socket has already been authenticated. Adding a second check here would
# only reintroduce the per-app secret this setup exists to avoid.



# --------------------------------------------------------------- api: meta ---
@app.route("/api/models")
def api_models():
    # background-refresh on first request, serve from cache otherwise
    if registry.last_refresh == 0.0:
        threading.Thread(target=registry.refresh, daemon=True).start()
    with db() as con:
        errs = {r["key"]: r["last_error"] for r in
                con.execute("SELECT key, last_error FROM providers")}
    return jsonify({"providers": {k: v["label"] for k, v in PROVIDERS.items()},
                    "models": registry.all_models(),
                    "errors": errs})


@app.route("/api/models/refresh", methods=["POST"])
def api_models_refresh():
    registry.refresh(force=True)
    return jsonify({"ok": True})


# -------------------------------------------------------------- api: chats ---
def _chat_row(con, cid):
    chat = con.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
    if not chat:
        return None, None
    msgs = con.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    return dict(chat), [dict(m) for m in msgs]


@app.route("/api/chats")
def api_chats_list():
    with db() as con:
        chats = con.execute(
            "SELECT * FROM chats ORDER BY updated_at DESC LIMIT 200").fetchall()
        out = []
        for c in chats:
            last = con.execute(
                "SELECT text FROM messages WHERE chat_id=? AND role='user' "
                "ORDER BY id DESC LIMIT 1", (c["id"],)).fetchone()
            out.append({"id": c["id"], "title": c["title"],
                        "models": json.loads(c["models"]),
                        "system": c["system"],
                        "temperature": c["temperature"],
                        "max_tokens": c["max_tokens"],
                        "updated_at": c["updated_at"],
                        "last_user": (last["text"][:120] if last else "")})
    return jsonify(out)


def msg_payload(m):
    return {"id": m["id"], "role": m["role"], "provider": m["provider"],
            "model_id": m["model_id"], "text": m["text"],
            "reasoning": m["reasoning"], "error": m["error"],
            "usage": json.loads(m["usage"]) if m["usage"] else None,
            "created_at": m["created_at"]}


@app.route("/api/chats", methods=["POST"])
def api_chats_create():
    body = request.get_json(silent=True) or {}
    now = int(time.time())
    with db() as con:
        cur = con.execute(
            "INSERT INTO chats(title, models, system, temperature, max_tokens,"
            " created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
            (body.get("title") or "new chat",
             json.dumps(body.get("models") or []),
             body.get("system") or "",
             body.get("temperature"), body.get("max_tokens"), now, now))
        cid = cur.lastrowid
    return jsonify({"id": cid})


@app.route("/api/chats/<int:cid>")
def api_chats_get(cid):
    with db() as con:
        chat, msgs = _chat_row(con, cid)
    if not chat:
        return jsonify({"error": "not found"}), 404
    chat["models"] = json.loads(chat["models"])
    chat["messages"] = [msg_payload(m) for m in msgs]
    return jsonify(chat)


@app.route("/api/chats/<int:cid>", methods=["PATCH"])
def api_chats_patch(cid):
    body = request.get_json(silent=True) or {}
    fields, vals = [], []
    for f in ("title", "system", "temperature", "max_tokens"):
        if f in body:
            fields.append(f + "=?")
            vals.append(body[f])
    if "models" in body:
        fields.append("models=?")
        vals.append(json.dumps(body["models"]))
    if fields:
        fields.append("updated_at=?")
        vals.append(int(time.time()))
        with db() as con:
            con.execute(f"UPDATE chats SET {', '.join(fields)} WHERE id=?",
                        vals + [cid])
    return jsonify({"ok": True})


@app.route("/api/chats/<int:cid>", methods=["DELETE"])
def api_chats_delete(cid):
    with db() as con:
        con.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
        con.execute("DELETE FROM chats WHERE id=?", (cid,))
    return jsonify({"ok": True})


# ---------------------------------------------------------- api: streaming ---
def sse(obj):
    return f"data:{json.dumps(obj, ensure_ascii=False)}\n\n"


def _clean_messages(history):
    """Provider-safe message list, truncated through the last user turn.

    Assistant rows generated *after* the final user message are sibling
    answers from the same multi-model turn (or a pane being retried) and
    must not pollute this model's own context.
    """
    out = []
    for m in history:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant" and m["text"] and not m["error"]:
            out.append({"role": "assistant", "content": m["text"]})
    while out and out[-1]["role"] != "user":
        out.pop()
    return out


def _provider_payload(pkey, model_id, messages, body):
    payload = {"model": model_id, "messages": messages}
    if PROVIDERS[pkey].get("stream", True):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("max_tokens"):
        payload["max_tokens"] = body["max_tokens"]
    return payload


@app.route("/api/chats/<int:cid>/stream", methods=["POST"])
def api_stream(cid):
    """Run one prompt against ONE model, streamed back as SSE.

    The frontend fans a generation out to N parallel streams (one per model),
    which keeps every worker thread short-lived and nginx config simple.
    """
    body = request.get_json(silent=True) or {}
    pkey = body.get("provider")
    model_id = body.get("model_id")
    if pkey not in PROVIDERS or not model_id:
        return jsonify({"error": "provider/model_id required"}), 400
    # model_id lands in an upstream URL path (and :generateAnswer for AQA):
    # charset + length + no ".." keeps it a model slug, not a path escape.
    # The registry list changes as providers add models, so a static
    # allowlist would false-reject; this shape check cannot.
    if (not isinstance(model_id, str) or len(model_id) > 160
            or ".." in model_id
            or not re.match(r"^[A-Za-z0-9_.\-/:]+$", model_id)):
        return jsonify({"error": "bad model_id"}), 400
    temp = body.get("temperature")
    if temp is not None:
        try:
            temp = float(temp)
        except (TypeError, ValueError):
            return jsonify({"error": "bad temperature"}), 400
        if not math.isfinite(temp) or not 0 <= temp <= 5:
            return jsonify({"error": "bad temperature"}), 400
        body["temperature"] = temp
    mtok = body.get("max_tokens")
    if mtok is not None:
        try:
            mtok = int(mtok)
        except (TypeError, ValueError):
            return jsonify({"error": "bad max_tokens"}), 400
        if not 1 <= mtok <= 1000000:
            return jsonify({"error": "bad max_tokens"}), 400
        body["max_tokens"] = mtok
    pconf = PROVIDERS[pkey]

    with db() as con:
        chat, msgs = _chat_row(con, cid)
    if not chat:
        return jsonify({"error": "chat not found"}), 404

    system = (body.get("system") if "system" in body else chat["system"]) or ""
    history = _clean_messages(msgs)
    messages = ([{"role": "system", "content": system}] if system else []) + history

    payload = _provider_payload(pkey, model_id, messages, body)
    headers = ModelRegistry._headers(pkey)
    headers["Content-Type"] = "application/json"

    def _post(url, payload, headers, pkey, **kw):
        pconf = PROVIDERS[pkey]
        if pconf.get("key_style") == "query":
            kw.setdefault("params", {})["key"] = pconf["api_key"]
            return HTTP.post(url, json=payload, headers=headers, **kw)
        return HTTP.post(url, json=payload, headers=headers, **kw)

    def _err_detail(r):
        try:
            j = r.json()
            return scrub((j.get("error") or {}).get("message") or json.dumps(j)[:400])
        except Exception:
            return scrub(r.text[:400])

    def _store(text, reasoning, usage, finish, error, latency):
        with db() as con:
            con.execute(
                "INSERT INTO messages(chat_id, role, provider, model_id,"
                " text, reasoning, usage, error, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, "assistant", pkey, model_id, text, reasoning,
                 json.dumps({"usage": usage, "finish": finish,
                             "latency_s": latency}),
                 error, int(time.time())))
            con.execute("UPDATE chats SET updated_at=? WHERE id=?",
                        (int(time.time()), cid))

    def relay():
        text_parts, reason_parts = [], []
        usage, finish, error = None, None, None
        aborted = False
        t0 = time.time()
        if pconf.get("kind") == "aqa":
            # Google attributed-QA: the conversation becomes the grounding
            # passage, the last user turn the question
            transcript = "\n".join(
                f"({m['role']}) {m['content']}" for m in messages)
            question = next((m["content"] for m in reversed(messages)
                             if m["role"] == "user"), transcript)
            aqa_payload = { "contents": [ { "parts": [ { "text": question } ] } ], "answerStyle": "ABSTRACTIVE", "inlinePassages": { "passages": [ { "id": "ctx", "content": { "parts": [ { "text": transcript[-8000:] } ] } } ] } }
            try:
                r = _post(f"{pconf['base_url']}/models/{model_id}:generateAnswer",
                          aqa_payload, {"Content-Type": "application/json"},
                          pkey, timeout=(15, 600))
                if r.status_code != 200:
                    error = f"HTTP {r.status_code}: {_err_detail(r)}"
                    yield sse({"t": "error", "error": error})
                else:
                    ans = r.json().get("answer", {})
                    parts = (ans.get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts)
                    finish = ans.get("finishReason")
                    if not text:
                        error = "empty reply"
                        yield sse({"t": "error", "error": error})
                    else:
                        text_parts.append(text)
                        yield sse({"t": "c", "d": text})
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {scrub(exc)}"
                yield sse({"t": "error", "error": error})
            finally:
                latency = round(time.time() - t0, 2)
                _store("".join(text_parts), "", usage, finish, error, latency)
                yield sse({"t": "done", "latency_s": latency, "usage": usage,
                           "finish": finish, "error": error})
            return
        if not pconf.get("stream", True):
            # non-streaming provider: single request, relay the whole
            # answer as one event so panes behave identically
            try:
                r = HTTP.post(pconf["base_url"] + "/chat/completions",
                              json=payload, headers=headers,
                              timeout=(15, 600))
                if r.status_code != 200:
                    # _err_detail scrubs provider keys (?key= style leaks into
                    # urllib3 error text); never forward raw detail.
                    error = f"HTTP {r.status_code}: {_err_detail(r)}"
                    yield sse({"t": "error", "error": error})
                else:
                    obj = r.json()
                    usage = obj.get("usage")
                    choice = (obj.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    text = msg.get("content") or ""
                    if not text:
                        # reasoning-heavy model spent the budget thinking;
                        # say so instead of showing a silently empty pane
                        reason_toks = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                        error = (f"empty reply (spent {reason_toks} reasoning tokens, "
                                 f"got no text)") if reason_toks else "empty reply"
                        yield sse({"t": "error", "error": error})
                    else:
                        text_parts.append(text)
                        yield sse({"t": "c", "d": text})
                        finish = choice.get("finish_reason")
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {scrub(exc)}"
                yield sse({"t": "error", "error": error})
            finally:
                latency = round(time.time() - t0, 2)
                text = "".join(text_parts)
                reasoning = "".join(reason_parts)
                with db() as con:
                    con.execute(
                        "INSERT INTO messages(chat_id, role, provider, model_id,"
                        " text, reasoning, usage, error, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, "assistant", pkey, model_id, text, reasoning,
                         json.dumps({"usage": usage, "finish": finish,
                                     "latency_s": latency}),
                         error, int(time.time())))
                    con.execute("UPDATE chats SET updated_at=? WHERE id=?",
                                (int(time.time()), cid))
                yield sse({"t": "done", "latency_s": latency, "usage": usage,
                           "finish": finish, "error": error})
            return
        try:
            with HTTP.post(pconf["base_url"] + "/chat/completions",
                           json=payload, headers=headers, stream=True,
                           timeout=(15, 600)) as r:
                if r.status_code != 200:
                    error = f"HTTP {r.status_code}: {_err_detail(r)}"
                    yield sse({"t": "error", "error": error})
                else:
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data:"):
                            continue
                        chunk = raw[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for ch in obj.get("choices", []):
                            delta = ch.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                text_parts.append(content)
                                yield sse({"t": "c", "d": content})
                            reason = delta.get("reasoning") or delta.get("reasoning_content")
                            if reason:
                                reason_parts.append(reason)
                                yield sse({"t": "r", "d": reason})
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {scrub(exc)}"
            yield sse({"t": "error", "error": error})
        except GeneratorExit:
            aborted = True
            error = error or "aborted by client"
            raise
        finally:
            latency = round(time.time() - t0, 2)
            text = "".join(text_parts)
            reasoning = "".join(reason_parts)
            with db() as con:
                con.execute(
                    "INSERT INTO messages(chat_id, role, provider, model_id,"
                    " text, reasoning, usage, error, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, "assistant", pkey, model_id, text, reasoning,
                     json.dumps({"usage": usage, "finish": finish,
                                 "latency_s": latency}),
                     error, int(time.time())))
                con.execute("UPDATE chats SET updated_at=? WHERE id=?",
                            (int(time.time()), cid))
            if not aborted:
                yield sse({"t": "done", "latency_s": latency, "usage": usage,
                           "finish": finish, "error": error})

    resp = Response(relay(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return resp


@app.route("/api/chats/<int:cid>/messages", methods=["POST"])
def api_append_user(cid):
    """Persist the user turn before the parallel streams start."""
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    with db() as con:
        chat = con.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
        if not chat:
            return jsonify({"error": "not found"}), 404
        cur = con.execute(
            "INSERT INTO messages(chat_id, role, provider, model_id, text,"
            " created_at) VALUES(?,?,?,?,?,?)",
            (cid, "user", "", "", text, int(time.time())))
        if chat["title"] in ("new chat", ""):
            con.execute("UPDATE chats SET title=?, updated_at=? WHERE id=?",
                        (text[:60], int(time.time()), cid))
        else:
            con.execute("UPDATE chats SET updated_at=? WHERE id=?",
                        (int(time.time()), cid))
    return jsonify({"id": cur.lastrowid})


@app.route("/api/messages/<int:mid>", methods=["DELETE"])
def api_message_delete(mid):
    """Drop an assistant row so a pane can be retried without leaving
    the stale answer in the history."""
    with db() as con:
        row = con.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        if row["role"] != "assistant":
            return jsonify({"error": "can only delete assistant messages"}), 400
        con.execute("DELETE FROM messages WHERE id=?", (mid,))
    return jsonify({"ok": True})


# ------------------------------------------------------------------- pages ---
@app.route("/")
def index():
    return render_template("index.html", prefix=PREFIX)


# ------------------------------------------------------------------ main ---
init_db()
threading.Thread(target=lambda: registry.refresh(force=True), daemon=True).start()

if __name__ == "__main__":
    log.info("llmtest listening on %s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
