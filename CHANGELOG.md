# Changelog

## 0.2.1 — 2026-11-08

- The vllm route defaults to `vllm-sr/d3-ultra` instead of `vllm-sr/d3-turbo`:
  vLLM Semantic Router serves `vllm-sr/d3-turbo` as `vllm-sr/d3-ultra` (same answers, the response names d3-ultra) and the key's quota lists only d3-ultra. `d3-turbo` stays a name for the route; the `VLLM_SR_MODEL` valve
  still picks another.

## 0.2.0 — 2026-10-24

### Features

- **`{effort}` in a tier's model id** is replaced by the decision model's
  effort (`cursor/claude-opus-5-5-{effort}` → `cursor/claude-opus-5-5-high`),
  for connections that list one model per effort. Such a model is sent no
  `reasoning_effort`.

### Chores

- The README says how to register the model when Open WebUI answers
  `Model not found` (`BYPASS_ADMIN_ACCESS_CONTROL` off).

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
