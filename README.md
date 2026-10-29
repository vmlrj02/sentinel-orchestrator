# Sentinel Orchestrator

A loan-underwriting agent you talk to on Telegram. Sentinel interviews the applicant in plain
language, scores the application with a credit-risk model, and explains the decision and
what would improve it.

```
Telegram ──► LangGraph agent ───────────────────────────────────────────► reply
              │
              ├─ sentinel   interview: extract fields from free text, ask for what's missing,
              │             show a summary and wait for confirmation
              ├─ scoring    POST /predict → logistic-regression credit model (FastAPI)
              └─ analyst    explain the decision + answer follow-up questions
```

- **Structured extraction**: one LLM call per turn returns the reply *and* any fields the user
  mentioned, with enum-constrained categoricals (`RENT/OWN/MORTGAGE/OTHER`, grades `A–G`, …).
- **Deterministic control flow**: routing, the confirmation summary and the derived
  `loan_percent_income` are plain code. The LLM never decides when to score.
- **Explainable scoring**: the model returns the top features pushing risk up and down
  (scaled value × coefficient), which the analyst turns into plain English.
- **Per-chat memory**: LangGraph checkpointer keyed by chat; `/start` begins a fresh application.

## Example

```
You:      Hi, I'm Priya
Sentinel: Thanks, Priya. How old are you?
You:      29, I earn 55000 a year
...
Sentinel: Here's your application:
          - Name: Priya
          - Age: 29
          - Annual income: 55,000
          ...
          Please confirm if these details are correct (Yes/No), or tell me what to change.
You:      yes
Sentinel: Good news, Priya, your loan has been approved. The model estimates a 15% default risk.
          What helped: education purpose, loan grade B ...
          What raised risk: loan size relative to income, renting ...
```

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. `cp .env.example .env` and fill in `TELEGRAM_TOKEN` and `OPENAI_API_KEY`.
3. Install and start:

```bash
uv sync
./run.sh          # starts the scoring API on :8000, then the bot
```

Message your bot on Telegram. Because the bot uses long polling, it works from your phone
anywhere without exposing any port. Your computer just needs to be on and running it.

### Keep it running (macOS)

```bash
scripts/install-service.sh              # start at login, restart on crash, stay awake on power
tail -f logs/sentinel.log
scripts/install-service.sh --uninstall
```

### Lock it to your own chat

The bot is reachable by anyone who finds its username, and each message costs OpenAI tokens.
Set `ALLOWED_CHAT_IDS` in `.env` to your chat ID (ask [@userinfobot](https://t.me/userinfobot)).

## Tests

```bash
uv run pytest
```

## Layout

```
config/settings.py        env config
src/main.py               entry point
src/bot/telegram_bot.py   Telegram handlers, per-chat sessions
src/graph/state.py        agent state + extraction schema
src/graph/nodes.py        sentinel / scoring / analyst nodes
src/graph/workflow.py     LangGraph wiring and routing
scoring/app.py            FastAPI credit-risk model service
scoring/model/            trained model, scaler, feature columns
```

The scoring service comes from
[creditRiskModel-logisticRegression](https://github.com/vmlrj02/creditRiskModel-logisticRegression),
with one fix: categorical one-hot encoding is no longer dropped for single-row requests.
