/* llmtest - multi-model side-by-side playground */
"use strict";

// Deploy prefix from a meta tag, not an inline script: the CSP has no
// 'unsafe-inline', so inline scripts do not run. Missing tag = root deploy.
const PREFIX = document.querySelector('meta[name="llmtest-prefix"]')?.content || "";
const API = (PREFIX + "/api").replace(/\/\/+api/, "/api");
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

// ---------------------------------------------------------------- state ---
const S = {
  providers: {},            // key -> label
  models: {},               // key -> [{id,label,free,group}]
  picked: new Set(),        // "provider::id"
  chats: [],
  chat: null,               // current chat object (with messages)
  tab: null,                // active provider tab in picker
  running: false,
  controllers: [],          // AbortControllers of live streams
};

const keyOf = (p, m) => p + "::" + m;
const splitKey = k => { const i = k.indexOf("::"); return [k.slice(0, i), k.slice(i + 2)]; };
const labelOf = (p, m) => (S.models[p] || []).find(x => x.id === m)?.label || m;

// ---------------------------------------------------------------- utils ---
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderMarkdown(el, text) {
  // marked parses; then strip script/event-handler junk from the DOM.
  // Stored model output is re-rendered on every chat open, so anything
  // missed here persists — err on the side of removing too much.
  el.innerHTML = marked.parse(text || "", { breaks: true, gfm: true });
  el.querySelectorAll(
    "script,iframe,object,embed,form,style,link,meta,base").forEach(n => n.remove());
  el.querySelectorAll("*").forEach(n => {
    for (const a of [...n.attributes]) {
      const an = a.name.toLowerCase();
      const av = (a.value || "").trim().toLowerCase();
      if (an.startsWith("on") ||
          an === "formaction" || an === "srcdoc" ||
          ((an === "href" || an === "xlink:href" || an === "src") &&
           (av.startsWith("javascript:") || av.startsWith("vbscript:") ||
            av.startsWith("data:text/html")))) {
        n.removeAttribute(a.name);
      }
    }
  });
}
function toast(msg, ms = 3000) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}
async function jfetch(url, opts) {
  const r = await fetch(url, opts);
  // A 401 here means the SSO session lapsed mid-session. Send the browser to
  // the login page; the gate hands it back to us afterwards.
  if (r.status === 401) { location.href = "https://sso.inbox.li/sso/login?next=" + encodeURIComponent(location.href); throw new Error("unauthorized"); }
  return r;
}

// ------------------------------------------------------------ model data ---
async function loadModels(refresh = false) {
  if (refresh) await jfetch(API + "/models/refresh", { method: "POST" });
  const r = await jfetch(API + "/models");
  const d = await r.json();
  S.providers = d.providers; S.models = d.models;
  for (const [p, e] of Object.entries(d.errors || {}))
    if (e) toast("model list failed for " + p + ": " + e, 5000);
  const n = Object.values(d.models).reduce((a, v) => a + v.length, 0);
  $("#model-count").textContent = n + " models";
  if (!S.tab) S.tab = Object.keys(S.providers)[0];
  drawPicker(); drawChips();
}

// -------------------------------------------------------------- picker UI ---
function drawPicker() {
  const tabs = $("#provider-tabs");
  tabs.innerHTML = "";
  for (const [k, label] of Object.entries(S.providers)) {
    const b = document.createElement("button");
    b.textContent = label + " (" + (S.models[k] || []).length + ")";
    b.className = k === S.tab ? "on" : "";
    b.onclick = () => { S.tab = k; drawPicker(); };
    tabs.appendChild(b);
  }
  const list = $("#mp-list");
  list.innerHTML = "";
  const q = $("#mp-search").value.toLowerCase();
  const models = (S.models[S.tab] || []).filter(m =>
    !q || m.id.toLowerCase().includes(q) || (m.label || "").toLowerCase().includes(q));
  const groups = {};
  for (const m of models) (groups[m.group || "other"] ||= []).push(m);
  for (const [g, ms] of Object.entries(groups)) {
    const h = document.createElement("div"); h.className = "mp-group";
    const span = document.createElement("span"); span.textContent = g;
    const sel = document.createElement("label"); sel.textContent = "select all";
    sel.onclick = () => {
      const allOn = ms.every(m => S.picked.has(keyOf(S.tab, m.id)));
      ms.forEach(m => allOn ? S.picked.delete(keyOf(S.tab, m.id))
                            : S.picked.add(keyOf(S.tab, m.id)));
      syncPicker();
    };
    h.append(span, sel); list.appendChild(h);
    for (const m of ms) {
      const row = document.createElement("label"); row.className = "mp-item";
      const cb = document.createElement("input"); cb.type = "checkbox";
      cb.dataset.key = keyOf(S.tab, m.id); cb.checked = S.picked.has(cb.dataset.key);
      const nm = document.createElement("span"); nm.textContent = m.label;
      const id = document.createElement("span"); id.className = "mp-id"; id.textContent = m.id;
      row.append(cb, nm);
      if (m.free) { const f = document.createElement("span"); f.className = "mp-free"; f.textContent = "free"; row.appendChild(f); }
      row.appendChild(id);
      cb.onchange = () => { cb.checked ? S.picked.add(cb.dataset.key) : S.picked.delete(cb.dataset.key); afterPickChange(); };
      list.appendChild(row);
    }
  }
}
function syncPicker() {
  $$("#mp-list input[type=checkbox]").forEach(cb =>
    cb.checked = S.picked.has(cb.dataset.key));
  afterPickChange();
}
function afterPickChange() {
  $("#picked-count").textContent = S.picked.size;
  drawChips();
  if (S.chat) persistModels();
}
function drawChips() {
  const box = $("#picked-chips"); box.innerHTML = "";
  for (const k of S.picked) {
    const [p, m] = splitKey(k);
    const c = document.createElement("span"); c.className = "chip";
    const pr = document.createElement("span"); pr.className = "prov"; pr.textContent = S.providers[p] || p;
    const nm = document.createElement("b"); nm.textContent = labelOf(p, m);
    const x = document.createElement("span"); x.className = "x"; x.textContent = "✕";
    x.onclick = () => { S.picked.delete(k); syncPicker(); };
    c.append(pr, nm, x); box.appendChild(c);
  }
}

// ---------------------------------------------------------------- chats ---
async function loadChats() {
  S.chats = await (await jfetch(API + "/chats")).json();
  const list = $("#chat-list"); list.innerHTML = "";
  for (const c of S.chats) {
    const el = document.createElement("div");
    el.className = "chat-item" + (S.chat && c.id === S.chat.id ? " active" : "");
    const t = document.createElement("span"); t.className = "ci-title";
    t.textContent = c.title || "untitled";
    const d = document.createElement("button"); d.className = "ci-del"; d.textContent = "✕";
    d.title = "delete chat";
    d.onclick = async e => {
      e.stopPropagation();
      if (!confirm("delete chat “" + c.title + "”?")) return;
      await jfetch(API + "/chats/" + c.id, { method: "DELETE" });
      if (S.chat && S.chat.id === c.id) { S.chat = null; renderChat(); }
      loadChats();
    };
    el.append(t, d);
    el.onclick = () => openChat(c.id);
    list.appendChild(el);
  }
}
async function openChat(id) {
  const chat = await (await jfetch(API + "/chats/" + id)).json();
  S.chat = chat;
  S.picked = new Set((chat.models || []).map(([p, m]) => keyOf(p, m)));
  $("#system").value = chat.system || "";
  $("#temperature").value = chat.temperature ?? "";
  $("#max_tokens").value = chat.max_tokens ?? "";
  syncPicker(); renderChat(); loadChats();
}
async function newChat() {
  const r = await jfetch(API + "/chats", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "new chat", models: [...S.picked].map(splitKey) }) });
  const { id } = await r.json();
  await openChat(id);
}
async function persistModels() {
  if (!S.chat) return;
  const models = [...S.picked].map(splitKey);
  S.chat.models = models;
  await jfetch(API + "/chats/" + S.chat.id, { method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models }) });
}

// ------------------------------------------------------------- render ---
function renderChat() {
  const box = $("#chat"); box.innerHTML = "";
  if (!S.chat) {
    box.innerHTML = '<div class="empty-hint">pick some models and enter a prompt.<br>' +
      'every selected model answers the same conversation side by side.</div>';
    return;
  }
  const msgs = S.chat.messages || [];
  let i = 0;
  while (i < msgs.length) {
    if (msgs[i].role === "user") {
      const turn = addTurn(msgs[i].text);
      i++;
      const paneObjs = [];
      while (i < msgs.length && msgs[i].role === "assistant") {
        paneObjs.push(makePane(turn.panes, msgs[i].provider, msgs[i].model_id));
        i++;
      }
      // match stored assistant rows to panes
      let k = 0;
      for (const m of msgs.slice(0, i).filter(x => x.role === "assistant")) {
        const p = paneObjs[k++]; if (!p) continue;
        p.msgId = m.id;
        if (m.error) setPaneError(p, m.error);
        renderMarkdown(p.md, m.text);
        if (m.reasoning) {
          p.reason.classList.remove("hidden");
          p.reasonPre.textContent = m.reasoning;
        }
        setPaneMeta(p, m.usage, m.text);
      }
    } else { i++; }  // orphan assistant (shouldn't happen)
  }
  box.scrollTop = box.scrollHeight;
}

function addTurn(userText) {
  const node = $("#tpl-turn").content.cloneNode(true);
  node.querySelector(".turn-user").textContent = userText;
  // grab references BEFORE appendChild: appending a DocumentFragment
  // empties it, so queries on the fragment afterwards return null
  const panes = node.querySelector(".panes");
  $("#chat").appendChild(node);
  return { panes };
}

function makePane(container, provider, modelId) {
  const node = $("#tpl-pane").content.cloneNode(true);
  const pane = node.querySelector(".pane");
  const p = {
    pane, provider, model: modelId,
    md: node.querySelector(".md"),
    meta: node.querySelector(".pane-meta"),
    cursor: node.querySelector(".cursor"),
    reason: node.querySelector(".pane-reasoning"),
    reasonPre: node.querySelector(".pane-reasoning pre"),
    retry: node.querySelector(".pane-retry"),
    text: "",
  };
  node.querySelector(".pane-model").textContent =
    (S.providers[provider] || provider) + " · " + labelOf(provider, modelId);
  node.querySelector(".pane-model").title = provider + "/" + modelId;
  node.querySelector(".pane-copy").onclick = () =>
    navigator.clipboard.writeText(p.text).then(() => toast("copied"));
  p.retry.onclick = async () => {
    // remove the stored answer for this pane so the retry is the fresh turn
    if (p.msgId) {
      await jfetch(API + "/messages/" + p.msgId, { method: "DELETE" });
      p.msgId = null;
    }
    streamInto(p);
  };
  container.appendChild(node);
  container.classList.toggle("single", container.children.length === 1);
  return p;
}

function setPaneMeta(p, usage, text) {
  const bits = [];
  if (usage?.latency_s) bits.push(usage.latency_s + "s");
  if (usage?.usage) {
    const u = usage.usage;
    if (u.prompt_tokens != null || u.completion_tokens != null)
      bits.push((u.prompt_tokens ?? "?") + "→" + (u.completion_tokens ?? "?") + " tok");
  }
  if (text?.length) bits.push(text.length + "ch");
  p.meta.textContent = bits.join(" · ");
}
function setPaneError(p, msg) {
  p.pane.classList.add("err");
  p.md.innerHTML = '<span style="color:var(--red)">⚠ ' + esc(msg) + "</span>";
  p.retry.classList.remove("hidden");
}

// ----------------------------------------------------------- generation ---
function currentSettings() {
  return {
    system: $("#system").value.trim(),
    temperature: $("#temperature").value === "" ? null : parseFloat($("#temperature").value),
    max_tokens: $("#max_tokens").value === "" ? null : parseInt($("#max_tokens").value, 10),
  };
}
async function run() {
  if (S.running) return;
  const promptText = $("#prompt").value.trim();
  if (!promptText) { toast("enter a prompt first"); return; }
  if (!S.picked.size) { toast("select at least one model"); return; }
  if (!S.chat) await newChat();
  // persist settings + system prompt
  const st = currentSettings();
  await jfetch(API + "/chats/" + S.chat.id, { method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(st) });
  S.chat.system = st.system;
  // store user turn
  await jfetch(API + "/chats/" + S.chat.id + "/messages", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: promptText }) });
  const history = (S.chat.messages ||= []);
  history.push({ role: "user", text: promptText });
  const turn = addTurn(promptText);
  $("#prompt").value = "";
  const panes = [...S.picked].map(splitKey).map(([p, m]) => makePane(turn.panes, p, m));
  S.running = true;
  $("#btn-stop").classList.remove("hidden");
  $("#btn-run").disabled = true;
  $("#chat").scrollTop = $("#chat").scrollHeight;

  await Promise.allSettled(panes.map(p => streamInto(p, history)));

  S.running = false;
  $("#btn-stop").classList.add("hidden");
  $("#btn-run").disabled = false;
  // refresh persisted messages (assistant rows are saved server-side)
  const chat = await (await jfetch(API + "/chats/" + S.chat.id)).json();
  S.chat = chat;
  loadChats();
}

async function streamInto(p) {
  p.text = ""; p.md.textContent = ""; p.pane.classList.remove("err");
  p.md.classList.add("pending-dot"); p.cursor.classList.remove("hidden");
  p.retry.classList.add("hidden");
  const st = currentSettings();
  const ctrl = new AbortController();
  S.controllers.push(ctrl);
  // the server assembles history from the persisted chat (up to the last
  // user turn), so we only tell it which model to run and with what params
  const body = {
    provider: p.provider, model_id: p.model,
    system: st.system, temperature: st.temperature, max_tokens: st.max_tokens,
  };
  try {
    const r = await jfetch(API + "/chats/" + S.chat.id + "/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: ctrl.signal });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      setPaneError(p, e.error || ("HTTP " + r.status));
    } else {
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl + 2);
          if (!line.startsWith("data:")) continue;
          let ev; try { ev = JSON.parse(line.slice(5)); } catch { continue; }
          handleEvent(p, ev);
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") setPaneError(p, "stopped");
    else setPaneError(p, String(e.message || e));
  } finally {
    p.md.classList.remove("pending-dot");
    p.cursor.classList.add("hidden");
    S.controllers = S.controllers.filter(c => c !== ctrl);
  }
}

function handleEvent(p, ev) {
  if (ev.t === "c") {
    p.text += ev.d;
    renderMarkdown(p.md, p.text);
    p.md.parentElement.scrollTop = p.md.parentElement.scrollHeight;
  } else if (ev.t === "r") {
    p.reason.classList.remove("hidden");
    p.reasonPre.textContent += ev.d;
  } else if (ev.t === "error") {
    setPaneError(p, ev.error);
  } else if (ev.t === "done") {
    setPaneMeta(p, { latency_s: ev.latency_s, usage: ev.usage }, p.text);
    if (ev.error && !p.text) setPaneError(p, ev.error);
  }
}

function stopAll() {
  S.controllers.forEach(c => c.abort());
  toast("stopping…");
}

// -------------------------------------------------------------- wire up ---
$("#btn-run").onclick = run;
$("#btn-stop").onclick = stopAll;
$("#btn-new").onclick = () => { S.chat = null; renderChat(); loadChats();
  $("#system").value = ""; $("#temperature").value = ""; $("#max_tokens").value = ""; };
$("#btn-models").onclick = e => { e.stopPropagation(); $("#model-picker").classList.toggle("hidden"); };
$("#btn-settings").onclick = () => $("#settings-panel").classList.toggle("hidden");
$("#mp-search").oninput = drawPicker;
$("#mp-none").onclick = () => { S.picked.clear(); syncPicker(); };
$("#mp-refresh").onclick = async () => { toast("refreshing model lists…");
  $("#mp-refresh").disabled = true; await loadModels(true); $("#mp-refresh").disabled = false; };
document.addEventListener("click", e => {
  if (!e.target.closest("#model-picker") && !e.target.closest("#btn-models"))
    $("#model-picker").classList.add("hidden");
});
$("#prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
});
["system", "temperature", "max_tokens"].forEach(id =>
  $("#" + id).addEventListener("change", () => {
    if (!S.chat) return;
    jfetch(API + "/chats/" + S.chat.id, { method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentSettings()) });
  }));

marked.setOptions?.({ breaks: true });
loadModels();
loadChats();
renderChat();
