# Eval history

How the numbers moved, and why. Each row is a full run of `python -m evals.run`
(23 scripted conversations, judged by gpt-5.4; "x3" = every case run three times to expose variance).
Interviewer and advisor: gpt-5.4-mini.

| # | Run | Judge pass | Completed | Field accuracy | What changed before this run |
|---|-----|-----------:|----------:|---------------:|------------------------------|
| 1 | 2026-10-29 23:49 (x1) | 15.6% | 87.0% | 98.4% | Baseline |
| 2 | 2026-10-29 23:52 (x2) | 51.8% | 95.7% | 99.4% | Credit model explanations grouped by feature; judge given policy passages; policy rubric only on policy turns |
| 3 | 2026-10-29 23:55 (x2) | 89.5% | 97.8% | 100% | Counterfactual (`max_auto_approved_loan`) computed in code; judge given the application; advisor prompt v4 |
| 4 | 2026-10-29 23:56 (x3) | 92.5% | 91.3% | 99.3% | Judge told what the counterfactual means; Rahul follow-up made an explicit what-if |
| 5 | 2026-10-29 23:59 (x3) | **94.0%** | **95.7%** | **99.6%** | Unambiguous "yes" to the summary confirmed in code, not by the LLM |

Run 5 (the committed report): 0 prompt-injection leaks, 0 scoring before confirmation, 100% expected
tool use and citations, 88% of runs pass every check, 74% of cases pass on all three repeats.

## What the failures taught us

1. **The credit model's explanations were misleading.** Contributions were reported per one-hot
   column, so a grade-B applicant was told `loan_grade_D` lowered their risk (it meant "not being
   grade D"). The advisor translated that into "your grade B helped", and the judge rightly failed it
   for contradicting the record. Fix: sum dummy columns back into their feature and label them with the
   applicant's own value (`loan_grade=B`). This was a bug in the model API, found by an LLM eval.
2. **Half the early judge failures were the harness, not the agent.** The judge had not been shown the
   application or the retrieved passages, so true statements looked invented. Always read failing cases
   before changing prompts.
3. **"Explain what would help" invited made-up advice.** Prompting alone did not stop it. Computing the
   counterfactual in code (largest loan this lender would auto-approve) gave the advisor a grounded fact
   and removed a tool round trip.
4. **Run-to-run variance is real.** Cases that failed once passed on rerun. Single runs of a small set
   cannot distinguish a 2-point change from noise, so comparisons use repeats.
5. **Crisp rules belong in code.** Recognising a plain "yes" to the summary was left to the LLM and
   occasionally missed; a strict regex now handles the unambiguous cases and the LLM handles the rest.

## Retrieval

| mode | recall@1 | recall@4 | MRR |
|------|---------:|---------:|----:|
| BM25 | 0.708 | 0.792 | 0.767 |
| dense | **0.833** | **1.000** | **0.896** |
| hybrid (RRF, BM25 weight 0.3) | 0.708 | 0.958 | 0.828 |

A BM25 weight sweep (0 to 1.0) found every non-zero weight lowered MRR on this corpus, so production
uses dense retrieval; see `docs/decisions/003-retrieval.md`.

## Known gaps

- `messy-priya` says "never missed anything", which is not the same as "never defaulted"; the bot
  sometimes asks again (arguably correct) and the script has no answer, so the case stalls.
- The judge has not yet been calibrated against human labels; the next step is labelling ~50 verdicts
  and measuring agreement before trusting it as a CI gate.
- 24 retrieval and 23 conversation cases are enough to catch regressions, not to rank close variants.
