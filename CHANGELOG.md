# Changelog

## 0.1.0 — 2026-10-23

### Features

- **Jev Router pipe for Open WebUI** (`jev_router.py`): one model in the model
  list that asks a decision model for a tier and an effort each turn, and hands
  the turn to the Open WebUI model named for that tier (`HAIKU_MODEL`,
  `SONNET_MODEL`, `OPUS_MODEL`, `FABLE_MODEL`).
- **Decision models:** `typesafe` (Jev direct), `openrouter` (Jev on OpenRouter,
  System One), `openrouter-decisions` (OpenRouter's Decisions API), `laya` (your
  own server) and `vllm` (vLLM Semantic Router's d3-turbo), alone or as a list
  tried in order within one time budget.
- **Effort as `reasoning_effort`**, mapped per level (`EFFORT_MAP`) and sent only
  to the tiers whose model takes it (`EFFORT_TIERS`).
- **Stickiness per chat**: a switch has to clear `STICKY_CONFIDENCE`, or a bar
  per decision model (`DECIDER_BARS`).
- **Background tasks** (titles, tags, follow-ups) go to `TASK_TIER` without a
  decision.
- **Status line** above each reply: tier, effort, decider and confidence, time,
  and the model the turn went to.
- `scripts/check_decider.py` asks the decision models outside Open WebUI.
