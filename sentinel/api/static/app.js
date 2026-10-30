"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};



const SUGGESTIONS = {
  new: ["Hi", "I'd like a personal loan", "What documents do I need?", "Is there a fee if I repay early?"],
  collecting: ["What documents do I need?", "Is there a fee to repay early?"],
  confirming: [],
  under_review: [],
  decided: ["Why did I get this decision?", "What if I borrowed less?", "How do I appeal?"],
};

const app = {
  lender: "aster",
  cid: store.get("sentinel.cid"),
  userId: store.get("sentinel.user") || crypto.randomUUID().slice(0, 12),
  busy: false,
  status: "new",
  messageCount: 0,
  pollTimer: null,
};
store.set("sentinel.user", app.userId);

/* ---------- Helpers ---------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) if (child !== null && child !== undefined) node.append(child);
  return node;
}
async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "content-type": "application/json", accept: "application/json" }, ...options });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

/* ---------- Thread rendering ---------- */

function scrollThread() { const t = $("#thread"); t.scrollTop = t.scrollHeight; }

function addUser(text) {
  $("#thread").append(el("div", { class: "msg user" }, el("div", { class: "bubble" }, text)));
  scrollThread();
}

function addBot(text = "", { error = false } = {}) {
  const bubble = el("div", { class: "bubble" });
  const msg = el("div", { class: `msg bot${error ? " error" : ""}` }, el("div", { class: "chips" }), bubble);
  if (text) renderText(bubble, text, []);
  $("#thread").append(msg);
  scrollThread();
  return msg;
}

function typingIndicator() { return el("span", { class: "typing", "aria-label": "Thinking" }, el("i"), el("i"), el("i")); }

/** Plain text with [tenant/doc#section] citations turned into numbered links to the sources below. */
function renderText(bubble, text, citations) {
  bubble.textContent = "";
  const ids = citations.map((c) => c.id);
  const parts = text.split(/\[([a-z0-9_-]+\/[a-z0-9_.-]+(?:#[a-z0-9-]+)?)\]/g);
  parts.forEach((part, i) => {
    if (i % 2 === 0) { bubble.append(part); return; }
    const n = ids.indexOf(part) + 1;
    if (n > 0) bubble.append(el("sup", {}, el("a", { href: "#", title: part, onclick: (e) => { e.preventDefault(); openCitation(bubble, n); } }, String(n))));
  });
}

function openCitation(bubble, n) {
  const details = bubble.parentElement.querySelectorAll("details.citation")[n - 1];
  if (details) { details.open = true; details.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
}

function addCitations(msg, citations) {
  citations.forEach((c, i) => {
    const node = $("#citation-tpl").content.firstElementChild.cloneNode(true);
    node.querySelector("summary").textContent = `${i + 1}. ${c.title} › ${c.section}`;
    node.querySelector("p").textContent = c.text.split("\n").slice(1).join("\n");
    msg.append(node);
  });
}

/** Quick replies for questions with fixed answers (home, purpose, grade, yes/no). */
function addOptions(msg, options) {
  if (!options?.length) return;
  document.querySelectorAll(".options button").forEach((b) => { b.disabled = true; });  // only the latest are live
  msg.append(el("div", { class: "options" }, options.map((o) => el("button", { type: "button", onclick: () => send(o) }, o))));
}

function addChip(msg, text, kind = "") {
  msg.querySelector(".chips").append(el("span", { class: `chip ${kind}` }, text));
}

function addMeta(msg, { traceId, traceUrl, feedback }) {
  const meta = el("div", { class: "meta" });
  if (traceUrl) meta.append(el("a", { href: traceUrl, target: "_blank", rel: "noopener" }, "View trace ↗"));
  if (feedback && traceId) {
    const vote = (rating, button) => async () => {
      meta.querySelectorAll("button").forEach((b) => b.classList.remove("chosen"));
      button.classList.add("chosen");
      await api("/v1/feedback", { method: "POST", body: JSON.stringify({ trace_id: traceId, rating }) }).catch(() => {});
    };
    const up = el("button", { title: "Helpful", "aria-label": "Helpful" }, "👍");
    const down = el("button", { title: "Not helpful", "aria-label": "Not helpful" }, "👎");
    up.addEventListener("click", vote(1, up));
    down.addEventListener("click", vote(-1, down));
    meta.append(up, down);
  }
  if (meta.childElementCount) msg.append(meta);
}


/* ---------- Sending a turn (SSE over fetch) ---------- */

async function ensureConversation() {
  if (app.cid) return app.cid;
  const { conversation_id } = await api("/v1/conversations", { method: "POST" });
  app.cid = conversation_id;
  store.set("sentinel.cid", app.cid);
  return app.cid;
}

async function send(text) {
  text = text.trim();
  if (!text || app.busy) return;
  app.busy = true;
  $("#send").disabled = true;
  document.querySelectorAll(".options button").forEach((b) => { b.disabled = true; });
  addUser(text);

  const cid = await ensureConversation();
  let msg = addBot();
  const bubble = () => msg.querySelector(".bubble");
  bubble().append(typingIndicator());
  let streamed = "";
  let citations = [];
  let lastNode = null;
  let lastOptions = [];
  let gotAnyMessage = false;

  try {
    const response = await fetch(`/v1/conversations/${encodeURIComponent(cid)}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify({ text, lender: app.lender, user_id: app.userId, client_message_id: crypto.randomUUID() }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split;
      while ((split = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const line = raw.split("\n").find((l) => l.startsWith("data: "));
        if (line) handle(JSON.parse(line.slice(6)));
      }
    }
  } catch (err) {
    msg.classList.add("error");
    renderText(bubble(), "Connection problem. Please try again.", []);
    console.error(err);
  } finally {
    app.busy = false;
    $("#send").disabled = false;
    $("#input").focus();
    await refreshState();
  }

  function handle(event) {
    switch (event.type) {
      case "status":
        addChip(msg, event.text);
        break;
      case "tool_call":
        addChip(msg, event.name === "search_policy" ? "Checked the lender's policy documents" : "Ran a what-if on the credit model", "tool");
        break;
      case "citations":
        citations = citations.concat(event.items.filter((c) => !citations.some((x) => x.id === c.id)));
        break;
      case "token":
        if (!streamed) bubble().textContent = "";
        streamed += event.text;
        bubble().textContent = streamed;
        scrollThread();
        break;
      case "message":
        if (gotAnyMessage && event.node !== "advisor") msg = addBot();
        gotAnyMessage = true;
        lastNode = event.node;
        lastOptions = event.options || [];
        renderText(bubble(), event.text, citations);
        if (event.node === "advisor" && citations.length) addCitations(msg, citations);
        scrollThread();
        break;
      case "done":
        if (!gotAnyMessage && streamed) renderText(bubble(), streamed, citations);
        addOptions(msg, lastOptions);
        addMeta(msg, { traceId: event.trace_id, traceUrl: event.trace_url, feedback: lastNode === "advisor" });
        scrollThread();
        break;
      case "error":
        msg.classList.add("error");
        renderText(bubble(), event.message, []);
        break;
    }
  }
}

/* ---------- Conversation state ---------- */

async function refreshState() {
  if (!app.cid) { renderState(null); return; }
  try {
    const state = await api(`/v1/conversations/${encodeURIComponent(app.cid)}`);
    // New messages while the application was under review can only be the underwriter's decision
    if (app.status === "under_review" && state.messages.length > app.messageCount && !app.busy) {
      for (const m of state.messages.slice(app.messageCount)) {
        if (m.role === "assistant") addOptions(addBot(m.text), m.options);
      }
    }
    app.messageCount = state.messages.length;
    renderState(state);
  } catch (err) { console.error(err); }
}

function renderState(state) {
  app.status = state?.messages.length ? state.status : "new";
  renderSuggestions();
  clearInterval(app.pollTimer);
  if (app.status === "under_review") app.pollTimer = setInterval(refreshState, 3000);
}

function renderSuggestions() {
  const box = $("#suggestions");
  box.textContent = "";
  for (const text of SUGGESTIONS[app.status] || []) {
    box.append(el("button", { type: "button", onclick: () => send(text) }, text));
  }
}

/* ---------- Conversation lifecycle ---------- */


function newConversation() {
  app.cid = null;
  app.messageCount = 0;
  store.set("sentinel.cid", "");
  $("#thread").textContent = "";
  renderState(null);
}


async function restore() {
  if (!app.cid) { renderState(null); return; }
  try {
    const state = await api(`/v1/conversations/${encodeURIComponent(app.cid)}`);
    if (!state.messages.length) { renderState(state); return; }
    state.messages.forEach((m, i) => {
      if (m.role === "user") { addUser(m.text); return; }
      const msg = addBot(m.text);
      if (i === state.messages.length - 1) addOptions(msg, m.options);
    });
    app.messageCount = state.messages.length;
    renderState(state);
  } catch { newConversation(); }
}

function autosize() {
  const input = $("#input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function boot() {
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; autosize(); send(t); });
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#input").addEventListener("input", autosize);
  $("#new-app").addEventListener("click", newConversation);
  await restore();
}

boot();
