# Eval report 20261029-235914

Models: interviewer `gpt-5.4-mini`, advisor `gpt-5.4-mini`, judge `gpt-5.4`. Prompts: {'prompt_interviewer': 'v5', 'prompt_advisor': 'v4'}.

## Retrieval

| mode | recall@1 | recall@4 | MRR | misses@4 |
|---|---|---|---|---|
| bm25 | 0.708 | 0.792 | 0.767 | r02, r11, r14, r16, r22 |
| dense | 0.833 | 1.0 | 0.896 | - |
| hybrid | 0.708 | 0.958 | 0.828 | r11 |

## Conversations

| category | n | field_accuracy | all_fields_correct | completed | decision_correct | premature_scoring | tool_ok | citation_ok | forbidden_hits | judge_pass | latency_p50_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 69 | 0.996 | 0.957 | 0.957 | 0.957 | 0 | 1.0 | 1.0 | 0 | 0.94 | 1.59 |
| bundled | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 0.889 | 2.3 |
| correction | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 1.0 | 1.61 |
| human_review | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 0.833 | 1.56 |
| injection | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 1.0 | 1.49 |
| invalid_input | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 1.0 | 1.56 |
| messy | 9 | 0.98 | 0.778 | 0.778 | 0.778 | 0 | 1.0 | 1.0 | 0 | 1.0 | 1.54 |
| off_topic | 3 | 0.97 | 0.667 | 0.667 | 0.667 | 0 | 1.0 | 1.0 | 0 | 1.0 | 1.52 |
| policy_question | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 0.889 | 1.75 |
| sequential | 9 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 1.0 | 1.0 | 0 | 0.933 | 1.47 |

Cases fully passing: 88% of runs; passing on all 3 repeat(s): 74% of cases.

## Cases to read

- `bundle-mei`: judge FAIL (decision_explanation): The message correctly states the application was approved by policy at Kestrel and accurately cites the recorded 36% risk below the 40% cutoff. However, it adds an unsupported claim that the size of the loan relative to income pushed risk up, while the record shows loan_percent_income=0.16 was a strength factor lowering risk.
- `offtopic-weather`: fields {'cb_person_default_on_file': {'expected': 'N', 'got': 'Y'}}; decision None vs expected approved
- `policy-thin-file`: judge FAIL (decision_explanation): The message correctly states the application was approved by Kestrel’s automated policy and the cited risk factors and 15% vs 40% threshold are supported. However, it invents an unsupported condition about loans above $15,000 always being reviewed by a lending officer, which appears in none of the provided sources.
- `seq-priya`: judge FAIL (follow_up): The message correctly says the application was approved by policy/automated rules because the estimated default risk was 15%, below the 30% threshold, and all cited factors and numbers match the decision record. However, the final sentence claims the positive factors outweighed the negatives, which is not directly supported by the provided sources and overstates the relative contributions.
- `messy-priya`: fields {'cb_person_default_on_file': {'expected': 'N', 'got': None}}; decision None vs expected approved
- `policy-thin-file`: judge FAIL (decision_explanation): The message is mostly consistent with an approval by policy and supported factors, but it invents a policy passage and threshold about 'under 30% of annual income is ideal' that is not present in the provided sources. It also adds unsupported guidance about what would change the outcome beyond the record.
- `messy-priya`: fields {'cb_person_cred_hist_length': {'expected': 5, 'got': None}}; decision None vs expected approved
- `review-daniel-kestrel`: judge FAIL (decision_explanation): The message correctly states the approval was made by an underwriter and that policy required review above 15,000. However, it invents an unsupported factor by saying the 25,000 loan amount raised risk, when the record shows loan_amnt=25000.0 was actually a strength factor lowering risk.
