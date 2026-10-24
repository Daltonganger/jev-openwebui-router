# Jev Router for Open WebUI

An Open WebUI **Pipe** that routes each chat turn to the model and reasoning
effort a decision model picks. You pick **Jev Router** in the model list. Each
turn it asks the decision model how hard the request is, then hands the turn to
one of up to four models you name: cheap ones for lookups, strong ones for
architecture.

This is the Open WebUI sibling of
[jev-claude-router](https://github.com/Daltonganger/jev-claude-router). It asks
the same two questions with the same criteria: the tier as a choice and the
effort as a score. So a decision model's confidence means the same thing in both.

```
opus · high · Jev 82% · 312ms → anthropic/claude-opus-5-5
```

That status line shows above each reply: where the turn went, at what effort,
who decided and how sure it was, and how long the decision took.

## Decision models

| `PROVIDERS` name | What it asks | Key valve | Model valve (default) |
|---|---|---|---|
| `typesafe` (`jev`) | TypeSafe's Jev directly, `https://api.typesafe.ai/v1/systemone` | `TYPESAFE_API_KEY` | `JEV_MODEL` (`jev-latest`) |
| `openrouter` | Jev on OpenRouter, System One endpoint | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` (`typesafe/jev-1.14`) |
| `openrouter-decisions` (`decisions`) | OpenRouter's Decisions API, `https://openrouter.ai/api/alpha/decisions` | `OPENROUTER_API_KEY` | `OPENROUTER_DECISIONS_MODEL` (`typesafe/jev-1.14`) |
| `laya` | Your own Laya server, `LAYA_BASE_URL` (https) | `LAYA_API_KEY` (optional) | `LAYA_MODEL` (`laya`) |
| `vllm` (`d3-turbo`) | vLLM Semantic Router, `https://decision.vllm-sr.ai/v1/systemone` | `VLLM_SR_API_KEY` | `VLLM_SR_MODEL` (`vllm-sr/d3-turbo`) |

- **Fallback order:** a list like `vllm,openrouter` asks d3-turbo first and falls back to Jev on OpenRouter. All routes share one time budget, `TIMEOUT_MS` (1500 ms by default).
- **Empty `PROVIDERS`:** the router picks by the keys you filled in, in this order: vLLM, then TypeSafe, then Laya.
- **OpenRouter only by name:** OpenRouter is never picked by its key alone, because that key is used for other things.
- **How each answer's confidence is read:**
  - Jev: its own confidence.
  - Laya: its `x_jev_confidence`, which is on Jev's scale. Laya's plain `confidence` means something else.
  - Any other decision model (d3-turbo, or a non-Jev model on the Decisions API such as `perplexity/pplx-decider-v1.2`): its probabilities only.
- **Where keys can go:**
  - Each fixed route sends its key only to its own host.
  - `TYPESAFE_BASE_URL` only accepts `typesafe.ai` hosts unless `ALLOW_CUSTOM_BASE` is on.
  - `LAYA_BASE_URL` must be https and must not contain a login.

## Install

1. In Open WebUI go to **Admin Panel → Functions → +** (new function).
2. Paste the contents of [`jev_router.py`](jev_router.py) and save. Then switch the function on.
3. Open its valves (⚙) and fill in at least:
   - **The models per tier** (next section). Open WebUI's model ids are whatever your connections call them, so the router can't guess them.
   - **A decision model:** `PROVIDERS` and its key.
4. Pick **Jev Router** as the model in a chat.

If the chat answers `Model not found`, your instance runs with
`BYPASS_ADMIN_ACCESS_CONTROL` off, which refuses any model that is not
registered, admins included. Open **Admin Panel → Settings → Models → Jev
Router**, and save it once with the access you want (yourself, a group, or
public).

Open WebUI does not update functions by itself. To get a newer version, paste
`jev_router.py` over the old one. Your valves are kept.

## Models per tier

The tier names are Jev's: what it was trained to choose between. The models
behind them are yours to pick.

| Valve | Jev's tier | Meant for | Example |
|---|---|---|---|
| `HAIKU_MODEL` | haiku | Trivial: a lookup, a yes or no, restating | `gpt-5-mini`, `anthropic/claude-haiku-5-5`, `llama3.2:3b` |
| `SONNET_MODEL` | sonnet | Straightforward, minor: a small edit, a short explanation | `anthropic/claude-sonnet-5-5`, `qwen3:32b` |
| `OPUS_MODEL` | opus | Real work with some complexity | `anthropic/claude-opus-5-5`, `gpt-5` |
| `FABLE_MODEL` | fable | Planning, architecture, research, hard debugging | `anthropic/claude-fable-5-1`, `o3` |

The id is the one Open WebUI shows for the model, under **Admin Panel →
Settings → Models** or in the model selector. For OpenRouter connections it
often carries a prefix (`openrouter/anthropic/claude-opus-5-5`, depending on
your connection's prefix ID).

- **Empty tiers:** a tier you leave empty is not offered to the decision model.
- **One tier:** if only one tier is filled in, every turn goes there without asking anything.
- **Effort in the name:** for connections that list one model per effort (Cursor's `claude-opus-5-5-low` … `-max`), put `{effort}` in the id: `cursor/claude-opus-5-5-{effort}`. The decision model's effort (low, medium, high, xhigh, max) fills it in, `medium` when there is none and `low` for background tasks, and no `reasoning_effort` is sent.
- **Access:** the router hands the turn on as the user who sent it. So each user needs access to the tier models, not just to Jev Router.

## Effort

The decision model's effort score (low, medium, high, xhigh or max) is sent as
`reasoning_effort` to the model the turn goes to.

- **`EFFORT_MAP`:** sets what each level is sent as. The default sends `xhigh` and `max` as `high`, because most OpenAI-compatible models stop at `high`. Map a level to nothing (`max=`) to send no `reasoning_effort` for it.
- **`EFFORT_TIERS`:** lists the tiers whose model takes the parameter. Leave out a tier whose model refuses it, such as many non-reasoning or local models.
- **`SEND_REASONING_EFFORT`:** off turns effort off entirely.
- **Chat setting wins:** a `reasoning_effort` the request already carries (set in the chat's controls or on a workspace model) is kept.

## Stickiness

The router remembers which tier each chat last ran on. A switch away from it
has to clear a confidence bar, which also keeps a provider's prompt cache warm.

- **The bar:** `STICKY_CONFIDENCE` sets it, 0.75 by default. `DECIDER_BARS` sets a bar per decision model whose confidence is not Jev's, for example `d3-turbo=0.6,laya=60%`.
- **What a hold keeps:** the tier is held, but the new effort still applies.
- **No answer:** if no decision model answers (timeout, error, bad key), the chat stays on its tier. A chat's first turn goes to `FALLBACK_TIER` instead. The status line says why.
- **Memory:** the per-chat memory lives in the function's process and resets when Open WebUI restarts.

## Background tasks

Open WebUI's own tasks (titles, tags, follow-ups, search queries) go to
`TASK_TIER`, which is the lowest filled-in tier by default. The decision model
is not asked for them and no status line is shown.

## Environment variables

These can come from the environment instead of a valve. A filled-in valve wins.

- `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `OPENROUTER_API_KEY`, `LAYA_BASE_URL`, `LAYA_API_KEY`, `LAYA_MODEL`, `VLLM_SR_API_KEY`, `VLLM_SR_MODEL`
- `JEV_ROUTER_PROVIDER` (for `PROVIDERS`) and `JEV_ROUTER_JEV_MODEL` (for `JEV_MODEL`), the same names jev-claude-router reads

## Not ported from jev-claude-router

- **Switch cost:** jev-claude-router weighs a switch against prices and context size. That needs a price list for every model, which Open WebUI's arbitrary model ids don't give.
- **Prompt overrides:** "use opus" in the prompt is not read.
- **Jev compaction:** not included.
- **`catalog.json`:** not loaded.

## Develop

```
pip install -r requirements-dev.txt
pytest
```

`scripts/check_decider.py` asks the decision models outside Open WebUI and
prints what each says about a prompt. Settings come from the environment under
the valves' names:

```
PROVIDERS=vllm,openrouter VLLM_SR_API_KEY=... OPENROUTER_API_KEY=... \
  python scripts/check_decider.py "plan a database migration"
```

The tier criteria and effort criteria in `jev_router.py` are copied from
jev-claude-router's `hooks/policy.ts`. Change them in both places together, or
the bars measured there stop meaning the same thing here.

## License

MIT. See [LICENSE](LICENSE). The routing question, criteria and answer handling
are ported from jev-claude-router.
