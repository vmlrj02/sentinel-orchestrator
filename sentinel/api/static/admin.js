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
// The token lives in the URL fragment (#...), which browsers never send to the server or log
const TOKEN = decodeURIComponent(location.hash.slice(1));
let lenders = [];

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: {
    "content-type": "application/json", accept: "application/json", authorization: `Bearer ${TOKEN}` } });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

/* ---------- Thread rendering ---------- */

/* ---------- Underwriter ---------- */

async function loadReviews() {
  const reviews = await api("/v1/reviews").catch(() => []);
  const badge = $("#review-count");
  badge.hidden = !reviews.length;
  badge.textContent = reviews.length;
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
      } catch (err) { e.target.closest(".review").append(el("p", { class: "muted small" }, `Could not submit: ${err.message}`)); }
      await Promise.all([loadReviews(), loadAudit()]);
    };
    const lender = lenders.find((l) => l.id === r.tenant_id);
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
      el("td", {}, lenders.find((l) => l.id === r.tenant_id)?.name ?? r.tenant_id),
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
  const e = await api("/v1/evals").catch(() => null);
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


async function boot() {
  if (!TOKEN) { document.body.replaceChildren(el("p", { class: "muted", style: "padding:2rem" }, "Not found.")); return; }
  try { lenders = await api("/v1/lenders"); } catch {
    document.body.replaceChildren(el("p", { class: "muted", style: "padding:2rem" }, "Not found.")); return;
  }
  $("#reviewer").value = store.get("sentinel.reviewer") || "admin";
  $("#reviewer").addEventListener("change", (e) => store.set("sentinel.reviewer", e.target.value));
  await Promise.all([loadReviews(), loadAudit(), loadSystem(), loadEvals()]);
  setInterval(() => { loadReviews(); loadSystem(); }, 5000);
  setInterval(loadAudit, 15000);
}

boot();
