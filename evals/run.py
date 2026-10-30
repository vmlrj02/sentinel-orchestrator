"""Run the eval suites against real models and write a report.

    uv run python -m evals.run                         # everything, default models
    uv run python -m evals.run --suite retrieval       # retrieval only (embeddings, no chat calls)
    uv run python -m evals.run --interviewer-model gpt-5.4-nano --tag nano-trial

Exit code is non-zero if any metric falls below evals/thresholds.json, so CI can gate prompt and
model changes on it.
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from sentinel.agent.state import REQUIRED_FIELDS
from sentinel.bootstrap import build_runtime
from sentinel.config import load_settings
from sentinel.prompts import prompt_versions
from sentinel.rag import Retriever
from sentinel.scoring.model import get_credit_model
from sentinel.service import ChatService
from sentinel.tenants import get_tenant

from .cases import CASES, Case

EVAL_DIR = Path(__file__).parent
REPORT_DIR = EVAL_DIR / "reports"
UI_SUMMARY = Path(__file__).parents[1] / "sentinel" / "api" / "static" / "eval-summary.json"


# --- Retrieval ------------------------------------------------------------------------------------

def run_retrieval(retriever: Retriever) -> dict[str, Any]:
    cases = [json.loads(line) for line in (EVAL_DIR / "retrieval_cases.jsonl").read_text().splitlines() if line]
    results: dict[str, Any] = {}
    for mode in ("bm25", "dense", "hybrid"):  # hybrid at the default BM25_WEIGHT
        ranks = []
        for case in cases:
            ids = [r.chunk.id for r in retriever.search(case["query"], case["lender"], k=10, mode=mode)]
            ranks.append(ids.index(case["expected"]) + 1 if case["expected"] in ids else None)
        results[mode] = {
            "recall@1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
            "recall@4": round(sum(r is not None and r <= 4 for r in ranks) / len(ranks), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
            "misses@4": [c["id"] for c, r in zip(cases, ranks, strict=True) if not r or r > 4],
        }
    return {"n": len(cases), "modes": results}


# --- LLM judge ------------------------------------------------------------------------------------

class Verdict(BaseModel):
    reason: str = Field(description="One or two sentences of reasoning, written before the verdict")
    passed: bool


EXPLANATION_RUBRIC = """You are grading a loan advisor's message to an applicant. PASS only if ALL hold:
1. It is consistent with the outcome ({decision}) and who decided it ({decided_by}). "policy" means the
   lender's automated rules, so "Kestrel approved it" is fine; saying a person decided when policy did,
   or the reverse, is a FAIL. A reply to a follow-up need not restate the outcome but must not contradict it.
2. Every factor, number or probability it mentions is supported by the APPLICATION, the DECISION RECORD,
   the WHAT-IF RESULTS or the POLICY PASSAGES below (simple arithmetic on them is fine). Factors in the
   record are written feature=value with a contribution: positive raised risk, negative lowered it.
   max_auto_approved_loan in the record was computed by re-scoring the application with the real model
   and policy: it is the largest loan the lender would approve automatically with everything else
   unchanged (null = no amount would). Statements based on it are supported.
3. It does not promise, hint at or invent a different outcome, condition or policy.
Minor omissions and reasonable paraphrases are fine; a contradiction or a fact found in none of the
sources is a FAIL.

APPLICANT'S MESSAGE: {question}

APPLICATION:
{application}

DECISION RECORD:
{record}

WHAT-IF RESULTS (hypothetical re-scores the advisor ran this turn):
{what_if}

POLICY PASSAGES (retrieved this turn):
{passages}

ADVISOR MESSAGE:
{answer}"""

POLICY_RUBRIC = """You are grading an answer to a loan applicant's policy question. PASS only if:
1. Every policy claim in the answer is supported by the PASSAGES (no outside knowledge, no invented numbers).
2. It addresses the question that was asked. (It may then continue the application by asking for the
   next detail; that is expected.)

QUESTION: {question}

PASSAGES:
{passages}

ANSWER:
{answer}"""


class Judge:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self._llm = ChatOpenAI(model=model, api_key=api_key, max_retries=3).with_structured_output(Verdict)  # type: ignore[arg-type]

    async def grade(self, rubric: str, **values: Any) -> Verdict:
        return await self._llm.ainvoke(rubric.format(**values))


# --- Conversations --------------------------------------------------------------------------------

def _match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and abs(actual - expected) <= max(0.011 * abs(expected), 0.01)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return bool(expected == actual)


def expected_decision(case: Case) -> str:
    truth = {**case.truth, "loan_percent_income": round(case.truth["loan_amnt"] / case.truth["person_income"], 2)}
    probability = get_credit_model().predict(truth)["probability"]
    return get_tenant(case.lender).decide(probability, truth)[0]


async def run_case(service: ChatService, retriever: Retriever, judge: Judge | None, case: Case,
                   run_id: str) -> dict[str, Any]:
    cid, user = f"eval-{run_id}-{case.id}", f"eval-{run_id}-{case.id}"
    tags = ["eval", f"case:{case.id}", f"category:{case.category}"]
    chunks = {c.id: c.text for c in retriever.chunks}
    replies: list[str] = []
    latencies: list[float] = []
    tools: set[str] = set()
    citations: set[str] = set()
    errors: list[str] = []
    judged: list[dict[str, Any]] = []
    scored_at: int | None = None
    assessment: dict[str, Any] | None = None
    explanation, explanation_what_if, trace_id = None, [], None

    transcript: list[dict[str, Any]] = []

    async def turn(text: str, index: int) -> Any:
        nonlocal scored_at, assessment, trace_id
        started = time.perf_counter()
        decided_before = assessment is not None
        result = await service.run_turn(cid, user, text, "eval", tenant_id=case.lender, tags=tags)
        latencies.append(time.perf_counter() - started)
        transcript.append({"user": text, "bot": result.messages, "tools": result.tools,
                           "citations": result.citations})
        replies.extend(result.messages)
        tools.update(result.tools)
        citations.update(result.citations)
        trace_id = result.trace_id or trace_id
        if result.error:
            errors.append(result.error)
        if result.assessment and scored_at is None:
            scored_at, assessment = index, result.assessment
        # Policy answers are graded while the application is in progress; later answers are graded
        # against the decision record by the explanation rubric
        if judge and not decided_before and not result.assessment and result.citations and result.messages:
            verdict = await judge.grade(POLICY_RUBRIC, question=text, answer=result.messages[-1],
                                        passages="\n\n".join(chunks.get(c, "") for c in result.citations))
            judged.append({"kind": "policy_answer", "turn": index, "passed": verdict.passed, "reason": verdict.reason})
        return result

    explanation_passages: list[str] = []
    for i, text in enumerate(case.turns):
        result = await turn(text, i)
        if scored_at == i and result.explained:
            explanation, explanation_what_if = result.messages[-1], result.what_if
            explanation_passages = result.citations

    state = await service.get_state(cid)
    review_decision = None
    if state["status"] == "under_review" and state["application_id"]:
        review_decision = "approved"
        reviewed = await service.submit_review(state["application_id"], review_decision, "eval-underwriter",
                                               "Eval: income verified")
        explanation, explanation_what_if = (reviewed.messages[-1] if reviewed.messages else None), reviewed.what_if
        explanation_passages = reviewed.citations
        tools.update(reviewed.tools)
        transcript.append({"user": "[underwriter approves]", "bot": reviewed.messages, "tools": reviewed.tools,
                           "citations": reviewed.citations})
        state = await service.get_state(cid)

    if judge and explanation and state["final_decision"]:
        final = state["final_decision"]
        verdict = await judge.grade(EXPLANATION_RUBRIC, decision=final["decision"], decided_by=final["decided_by"],
                                    record=json.dumps({"final": final, "assessment": state["assessment"]}),
                                    application=json.dumps(state["application"]),
                                    what_if=json.dumps(explanation_what_if) if explanation_what_if else "none",
                                    passages="\n\n".join(chunks.get(c, "") for c in explanation_passages) or "none",
                                    question=case.turns[-1], answer=explanation)
        judged.append({"kind": "decision_explanation", "passed": verdict.passed, "reason": verdict.reason})

    if case.follow_up and state["status"] == "decided":
        result = await turn(case.follow_up, len(case.turns))
        if judge and result.messages:
            final = state["final_decision"]
            verdict = await judge.grade(EXPLANATION_RUBRIC, decision=final["decision"],
                                        decided_by=final["decided_by"],
                                        record=json.dumps({"final": final, "assessment": state["assessment"]}),
                                    application=json.dumps(state["application"]),
                                        what_if=json.dumps(result.what_if) if result.what_if else "none",
                                        passages="\n\n".join(chunks.get(c, "") for c in result.citations) or "none",
                                        question=case.follow_up, answer=result.messages[-1])
            judged.append({"kind": "follow_up", "passed": verdict.passed, "reason": verdict.reason})

    application = state["application"]
    field_hits = {f: _match(case.truth[f], application.get(f)) for f in REQUIRED_FIELDS}
    expected = expected_decision(case)
    reply_text = "\n".join(replies).lower()
    outcome = {
        "id": case.id, "category": case.category, "lender": case.lender, "turns": len(case.turns),
        "field_accuracy": round(sum(field_hits.values()) / len(field_hits), 3),
        "wrong_fields": {f: {"expected": case.truth[f], "got": application.get(f)}
                         for f, ok in field_hits.items() if not ok},
        "completed": assessment is not None,
        "premature_scoring": scored_at is not None and scored_at < len(case.turns) - 1,
        "expected_decision": expected,
        "policy_decision": assessment["policy_decision"] if assessment else None,
        "decision_correct": bool(assessment) and assessment["policy_decision"] == expected,
        "reviewed": review_decision is not None,
        "tool_ok": case.expect_tool is None or case.expect_tool in tools,
        "citation_ok": case.expect_citation is None or case.expect_citation in citations,
        "forbidden_hits": [f for f in case.forbidden if f.lower() in reply_text],
        "judged": judged,
        "errors": errors,
        "latency_s": {"p50": round(statistics.median(latencies), 2), "max": round(max(latencies), 2)},
        "trace_id": trace_id,
        "tools": sorted(tools),
        "transcript": transcript,
    }
    if trace_id:
        service.tracer.score(trace_id, "eval_field_accuracy", outcome["field_accuracy"])
        service.tracer.score(trace_id, "eval_decision_correct", float(outcome["decision_correct"]))
    return outcome


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        verdicts = [j["passed"] for r in rows for j in r["judged"]]
        return {
            "n": len(rows),
            "field_accuracy": round(statistics.mean(r["field_accuracy"] for r in rows), 3),
            "all_fields_correct": round(sum(not r["wrong_fields"] for r in rows) / len(rows), 3),
            "completed": round(sum(r["completed"] for r in rows) / len(rows), 3),
            "decision_correct": round(sum(r["decision_correct"] for r in rows) / len(rows), 3),
            "premature_scoring": sum(r["premature_scoring"] for r in rows),
            "tool_ok": round(sum(r["tool_ok"] for r in rows) / len(rows), 3),
            "citation_ok": round(sum(r["citation_ok"] for r in rows) / len(rows), 3),
            "forbidden_hits": sum(len(r["forbidden_hits"]) for r in rows),
            "judge_pass": round(sum(verdicts) / len(verdicts), 3) if verdicts else None,
            "judged": len(verdicts),
            "errors": sum(len(r["errors"]) for r in rows),
            "latency_p50_s": round(statistics.median(r["latency_s"]["p50"] for r in rows), 2),
        }

    def clean(r: dict[str, Any]) -> bool:
        return (not r["wrong_fields"] and r["decision_correct"] and not r["forbidden_hits"] and r["tool_ok"]
                and r["citation_ok"] and all(j["passed"] for j in r["judged"]) and not r["errors"])

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
        by_case[r["id"]].append(clean(r))
    overall = agg(results)
    overall["case_pass_rate"] = round(sum(clean(r) for r in results) / len(results), 3)
    overall["consistent_cases"] = round(sum(all(v) for v in by_case.values()) / len(by_case), 3)
    overall["repeats"] = max(len(v) for v in by_case.values())
    return {"overall": overall, "by_category": {c: agg(rows) for c, rows in sorted(by_category.items())}}


def check_thresholds(report: dict[str, Any]) -> list[str]:
    thresholds = json.loads((EVAL_DIR / "thresholds.json").read_text())
    failures = []
    overall = report.get("conversations", {}).get("summary", {}).get("overall", {})
    for metric, minimum in thresholds.get("conversations", {}).items():
        value = overall.get(metric)
        if value is not None and (value > minimum if metric in ("premature_scoring", "forbidden_hits", "errors")
                                  else value < minimum):
            failures.append(f"conversations.{metric} = {value} (threshold {minimum})")
    production_mode = load_settings().retrieval_mode
    chosen = report.get("retrieval", {}).get("modes", {}).get(production_mode, {})
    for metric, minimum in thresholds.get("retrieval", {}).items():
        if metric in chosen and chosen[metric] < minimum:
            failures.append(f"retrieval.{production_mode}.{metric} = {chosen[metric]} (threshold {minimum})")
    return failures


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Eval report {report['run_id']}", "",
             f"Models: interviewer `{report['models']['interviewer']}`, advisor `{report['models']['advisor']}`, "
             f"judge `{report['models'].get('judge')}`. Prompts: {report['prompt_versions']}.", ""]
    if "retrieval" in report:
        lines += ["## Retrieval", "", "| mode | recall@1 | recall@4 | MRR | misses@4 |", "|---|---|---|---|---|"]
        for mode, m in report["retrieval"]["modes"].items():
            misses = ", ".join(m["misses@4"]) or "-"
            lines.append(f"| {mode} | {m['recall@1']} | {m['recall@4']} | {m['mrr']} | {misses} |")
        lines.append("")
    if "conversations" in report:
        cols = ["n", "field_accuracy", "all_fields_correct", "completed", "decision_correct", "premature_scoring",
                "tool_ok", "citation_ok", "forbidden_hits", "judge_pass", "latency_p50_s"]
        lines += ["## Conversations", "", "| category | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        summary = report["conversations"]["summary"]
        for name, row in [("**overall**", summary["overall"]), *summary["by_category"].items()]:
            lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
        o = summary["overall"]
        lines += ["", f"Cases fully passing: {o['case_pass_rate']:.0%} of runs; passing on all {o['repeats']} "
                      f"repeat(s): {o['consistent_cases']:.0%} of cases."]
        failing = [r for r in report["conversations"]["cases"]
                   if r["wrong_fields"] or not r["decision_correct"] or r["forbidden_hits"]
                   or any(not j["passed"] for j in r["judged"]) or not r["tool_ok"] or not r["citation_ok"]]
        if failing:
            lines += ["", "## Cases to read", ""]
            for r in failing:
                notes = []
                if r["wrong_fields"]:
                    notes.append(f"fields {r['wrong_fields']}")
                if not r["decision_correct"]:
                    notes.append(f"decision {r['policy_decision']} vs expected {r['expected_decision']}")
                notes += [f"judge FAIL ({j['kind']}): {j['reason']}" for j in r["judged"] if not j["passed"]]
                notes += [f"forbidden text: {r['forbidden_hits']}"] if r["forbidden_hits"] else []
                notes += ["expected tool not used"] if not r["tool_ok"] else []
                notes += ["expected passage not cited"] if not r["citation_ok"] else []
                lines.append(f"- `{r['id']}`: " + "; ".join(notes))
    if report.get("threshold_failures"):
        lines += ["", "## Threshold failures", "", *[f"- {f}" for f in report["threshold_failures"]]]
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", "retrieval", "conversations"], default="all")
    parser.add_argument("--interviewer-model")
    parser.add_argument("--advisor-model")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--only", help="comma-separated case ids or categories")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1, help="run each case N times to measure variance")
    parser.add_argument("--tag", default="", help="label for the report file name")
    args = parser.parse_args()

    settings = load_settings()
    settings = dataclasses.replace(
        settings, checkpointer="memory", rate_limit_per_minute=10_000, log_level="WARNING", public_traces=False,
        interviewer_model=args.interviewer_model or settings.interviewer_model,
        advisor_model=args.advisor_model or settings.advisor_model)
    logging.basicConfig(level="WARNING")
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report: dict[str, Any] = {
        "run_id": run_id, "prompt_versions": prompt_versions(),
        "models": {"interviewer": settings.interviewer_model, "advisor": settings.advisor_model,
                   "judge": None if args.no_judge else args.judge_model},
    }

    async with build_runtime(settings) as runtime:
        logging.getLogger().setLevel("WARNING")
        if args.suite in ("all", "retrieval"):
            report["retrieval"] = run_retrieval(runtime.retriever)
            print(json.dumps(report["retrieval"]["modes"], indent=1))
        if args.suite in ("all", "conversations"):
            only = set(args.only.split(",")) if args.only else None
            cases = [c for c in CASES if not only or c.id in only or c.category in only]
            judge = None if args.no_judge else Judge(args.judge_model, settings.openai_api_key)
            slots = asyncio.Semaphore(args.concurrency)
            started = time.perf_counter()

            async def bounded(case: Case, repeat: int) -> dict[str, Any]:
                async with slots:
                    outcome = await run_case(runtime.service, runtime.retriever, judge, case, f"{run_id}-{repeat}")
                    outcome["repeat"] = repeat
                    print(f"  {case.id:24s} fields={outcome['field_accuracy']:.2f} decision_ok="
                          f"{outcome['decision_correct']} judge={[j['passed'] for j in outcome['judged']]}")
                    return outcome

            results = await asyncio.gather(*(bounded(c, r) for r in range(args.repeats) for c in cases))
            report["conversations"] = {"cases": results, "summary": summarise(results),
                                       "wall_time_s": round(time.perf_counter() - started, 1)}

    report["threshold_failures"] = check_thresholds(report)
    REPORT_DIR.mkdir(exist_ok=True)
    name = f"{run_id}-{settings.interviewer_model}-{settings.advisor_model}{'-' + args.tag if args.tag else ''}"
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(report, indent=1, default=str))
    (REPORT_DIR / "latest.md").write_text(markdown(report))
    if args.suite == "all" and not args.only:
        UI_SUMMARY.write_text(json.dumps({
            "run_id": run_id, "models": report["models"], "prompt_versions": report["prompt_versions"],
            "retrieval": {m: {k: v for k, v in r.items() if k != "misses@4"}
                          for m, r in report["retrieval"]["modes"].items()},
            "conversations": report["conversations"]["summary"],
        }, indent=1))
    print("\n" + markdown(report))
    return 1 if report["threshold_failures"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
