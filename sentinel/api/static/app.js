"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

const FIELD_LABELS = {
  person_name: "Name", person_age: "Age", person_income: "Annual income", person_emp_length: "Years at job",
  person_home_ownership: "Home", loan_amnt: "Loan amount", loan_int_rate: "Interest rate", loan_intent: "Purpose",
  loan_grade: "Loan grade", cb_person_cred_hist_length: "Credit history", cb_person_default_on_file: "Previous default",
};


const SUGGESTIONS = {
  new: ["Hi", "I'd like a personal loan", "What documents do I need?", "Is there a fee if I repay early?"],
  collecting: ["What documents do I need?", "Is there a fee to repay early?"],
  confirming: [],
  under_review: [],
  decided: ["Why did I get this decision?", "What if I borrowed less?", "How do I appeal?"],
};

const app = {
  lenders: [],
  lender: store.get("sentinel.lender") || "aster",
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
const fmtMoney = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
/** Percent with one decimal when needed: 0.996 shows as 99.6%, never rounded up to 100%. */
const pct = (p) => { const tenths = Math.round(p * 1000) / 10; return `${tenths === 100 && p < 1 ? 99.9 : tenths}%`; };
function fmtField(key, value) {
  if (value === null || value === undefined) return null;
  if (key === "person_income" || key === "loan_amnt") return fmtMoney(value);
  if (key === "loan_int_rate") return `${value}%`;
  if (key === "person_emp_length" || key === "cb_person_cred_hist_length") return `${value} yr`;
  if (key === "cb_person_default_on_file") return value === "Y" ? "Yes" : "No";
  if (typeof value === "string" && value === value.toUpperCase()) return value.charAt(0) + value.slice(1).toLowerCase();
  return String(value);
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

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/* ---------- Sending a turn (SSE over fetch) ---------- */

async function ensureConversation() {
  if (app.cid) return app.cid;
  const { conversation_id } = await api("/v1/conversations", { method: "POST" });
  app.cid = conversation_id;
  store.set("sentinel.cid", app.cid);
  store.set("sentinel.lender", app.lender);
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
      case "assessment": {
        const label = event.policy_decision === "referred" ? "Referred to an underwriter" : event.policy_decision;
        msg.insertBefore(el("div", { class: `decision-line ${event.policy_decision}` },
          `${label} · estimated default risk ${pct(event.model.probability)}`), msg.querySelector(".bubble"));
        break;
      }
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
        if (m.role === "assistant") addChip(addBot(m.text), "An underwriter has decided", "tool");
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
  if (app.status === "under_review") {
    box.append(el("button", { type: "button", onclick: () => switchTab("underwriter") }, "Decide it yourself in the Underwriter tab →"));
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

function setLender(id) {
  app.lender = id;
  $("#lender").value = id;
  store.set("sentinel.lender", id);
}

async function restore() {
  if (!app.cid) { renderState(null); return; }
  try {
    const state = await api(`/v1/conversations/${encodeURIComponent(app.cid)}`);
    if (!state.messages.length) { renderState(state); return; }
    setLender(state.lender.id);
    state.messages.forEach((m, i) => {
      if (m.role === "user") { addUser(m.text); return; }
      const msg = addBot(m.text);
      if (i === state.messages.length - 1) addOptions(msg, m.options);
    });
    app.messageCount = state.messages.length;
    renderState(state);
  } catch { newConversation(); }
}

/* ---------- Underwriter ---------- */

async function loadReviews() {
  const reviews = await api("/v1/reviews").catch(() => []);
  const badge = $("#review-count");
  badge.hidden = !reviews.length;
  badge.textContent = reviews.length;
  if ($("#tab-underwriter").hidden) return;
  const queue = $("#queue");
  queue.textContent = "";
  if (!reviews.length) {
    queue.append(el("div", { class: "empty-state" }, "No applications waiting. Borderline applications appear here, e.g. try: “Asha, 29, 55000 a year, 4 years in my job, renting, 18000 for medical treatment at 13%, grade C, 5 years credit history, no defaults”."));
  }
  for (const r of reviews) {
    const note = el("textarea", { placeholder: "Note for the applicant and the audit log (e.g. income verified with payslips)" });
    const decide = (decision) => async (e) => {
      e.target.closest(".review").querySelectorAll("button").forEach((b) => { b.disabled = true; });
      try {
        await api(`/v1/reviews/${r.application_id}`, { method: "POST",
          body: JSON.stringify({ decision, reviewer: $("#reviewer").value || "demo-underwriter", note: note.value || null }) });
      } catch (err) { alert(`Could not submit: ${err.message}`); }
      await Promise.all([loadReviews(), loadAudit()]);
    };
    const lender = app.lenders.find((l) => l.id === r.tenant_id);
    queue.append(el("div", { class: "card review" },
      el("header", {}, el("strong", {}, r.application_id), el("span", { class: "muted small" }, lender?.name ?? r.tenant_id)),
      el("div", { class: "reason" }, `Referred: ${r.policy_reason}`),
      el("dl", { class: "fields" }, Object.entries(FIELD_LABELS).filter(([k]) => k !== "person_name").flatMap(([k, label]) => [
        el("dt", {}, label), el("dd", {}, fmtField(k, r.application[k]) ?? "—")])),
      note,
      el("div", { class: "actions" },
        el("button", { class: "approve", onclick: decide("approved") }, "Approve"),
        el("button", { class: "decline", onclick: decide("declined") }, "Decline")),
      el("div", { class: "muted small" }, `Opened ${new Date(r.created_at).toLocaleString()} · model ${r.model_version}`),
    ));
  }
}

async function loadAudit() {
  const rows = await api("/v1/applications?limit=25").catch(() => []);
  const body = $("#audit tbody");
  body.textContent = "";
  if (!rows.length) body.append(el("tr", {}, el("td", { colspan: 7, class: "muted" }, "No decisions yet.")));
  for (const r of rows) {
    body.append(el("tr", {},
      el("td", {}, r.application_id),
      el("td", {}, app.lenders.find((l) => l.id === r.tenant_id)?.name ?? r.tenant_id),
      el("td", {}, el("span", { class: `tag ${r.status}` }, r.status.replace("_", " "))),
      el("td", {}, r.decided_by),
      el("td", {}, pct(r.model_result.probability)),
      el("td", { class: "small" }, `${r.model_version} · ${Object.values(r.prompt_versions).join("/")}`),
      el("td", {}, r.trace_id ? el("code", { class: "small", title: r.trace_id }, r.trace_id.slice(0, 8)) : "—"),
    ));
  }
}

/* ---------- System ---------- */

function stat(label, value, sub = "") {
  return el("div", { class: "stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value), el("div", { class: "sub" }, sub));
}

async function loadSystem() {
  if ($("#tab-system").hidden) return;
  const s = await api("/v1/system").catch(() => null);
  if (!s) return;
  $("#sys-models").textContent = `Models: interviewer ${s.models.interviewer}, advisor ${s.models.advisor} · env ${s.environment} · Langfuse ${s.langfuse ? "on" : "off"}`;
  const t = s.turns;
  const decisions = Object.entries(s.decisions).map(([k, v]) => `${k.replace("underwriter_", "UW ")} ${v}`).join(" · ") || "none yet";
  $("#stats").replaceChildren(
    stat("Chat turns", t.total, `${t.ok || 0} ok · ${t.error || 0} errors · ${t.timeout || 0} timeouts`),
    stat("Turn latency p50", s.latency_s.p50 !== null ? `${s.latency_s.p50}s` : "—", "histogram bucket upper bound"),
    stat("Turn latency p95", s.latency_s.p95 !== null ? `${s.latency_s.p95}s` : "—", "end to end, incl. tools"),
    stat("In flight", s.in_flight, `cap ${s.max_concurrent_turns} concurrent turns`),
    stat("LLM calls", s.llm.calls, `${s.llm.errors} errors (retried, then fallback model)`),
    stat("Tokens per turn", s.llm.tokens_per_turn ?? "—", `${fmtMoney((s.llm.tokens.input || 0) + (s.llm.tokens.output || 0))} total`),
    stat("Decisions", Object.values(s.decisions).reduce((a, b) => a + b, 0), decisions),
    stat("Feedback", `${s.feedback.up || 0} 👍 ${s.feedback.down || 0} 👎`, "scored on the Langfuse trace"),
    stat("Rate limited", s.rate_limited, "per-user sliding window"),
  );
}

async function loadEvals() {
  const e = await fetch("/static/eval-summary.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const box = $("#evals");
  if (!e) { box.append(el("p", { class: "muted" }, "No eval run published yet.")); return; }
  const o = e.conversations.overall;
  box.replaceChildren(
    el("div", { class: "eval-grid" },
      stat("Field extraction", pct(o.field_accuracy), `${o.n} scripted conversations`),
      stat("Correct decisions", pct(o.decision_correct), "vs model + policy on ground truth"),
      stat("Judge: grounded answers", pct(o.judge_pass), `${o.judged} explanations graded`),
      stat("Injection leaks", o.forbidden_hits, "prompt or decision leaks"),
      stat("Scored before confirm", o.premature_scoring, "must be zero"),
      stat("Retrieval recall@4", pct(e.retrieval.dense["recall@4"]), `MRR ${e.retrieval.dense.mrr} (dense)`),
    ),
    el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Category", "Runs", "Field acc.", "Completed", "Decision ok", "Tool ok", "Citations ok", "Judge pass", "p50 latency"].map((h) => el("th", {}, h)))),
      el("tbody", {}, Object.entries(e.conversations.by_category).map(([name, r]) => el("tr", {},
        el("td", {}, name.replace("_", " ")), el("td", {}, r.n), el("td", {}, pct(r.field_accuracy)), el("td", {}, pct(r.completed)),
        el("td", {}, pct(r.decision_correct)), el("td", {}, pct(r.tool_ok)), el("td", {}, pct(r.citation_ok)),
        el("td", {}, r.judge_pass === null ? "—" : pct(r.judge_pass)), el("td", {}, `${r.latency_p50_s}s`)))),
    )),
    el("p", { class: "muted small" }, `Run ${e.run_id} · interviewer ${e.models.interviewer}, advisor ${e.models.advisor}, judge ${e.models.judge} · prompts ${Object.values(e.prompt_versions).join(", ")}`),
  );
}

/* ---------- Tabs & boot ---------- */

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  for (const id of ["apply", "underwriter", "system"]) $(`#tab-${id}`).hidden = id !== name;
  if (name === "underwriter") { loadReviews(); loadAudit(); }
  if (name === "system") loadSystem();
  if (name === "apply") refreshState();
}

function autosize() {
  const input = $("#input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function boot() {
  app.lenders = await api("/v1/lenders").catch(() => []);
  const select = $("#lender");
  for (const l of app.lenders) select.append(el("option", { value: l.id }, l.name));
  if (!app.lenders.some((l) => l.id === app.lender)) app.lender = app.lenders[0]?.id ?? "aster";
  setLender(app.lender);
  select.addEventListener("change", () => { setLender(select.value); newConversation(); });

  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; autosize(); send(t); });
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#input").addEventListener("input", autosize);
  $("#new-app").addEventListener("click", newConversation);
  $("#reviewer").value = store.get("sentinel.reviewer") || "demo-underwriter";
  $("#reviewer").addEventListener("change", (e) => store.set("sentinel.reviewer", e.target.value));

  await restore();
  loadEvals();
  loadReviews();
  setInterval(loadReviews, 5000);
  setInterval(loadSystem, 5000);
}

boot();
