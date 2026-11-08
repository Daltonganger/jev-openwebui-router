"""
title: Jev Router
author: Daltonganger
author_url: https://github.com/Daltonganger/jev-openwebui-router
description: Routes each chat turn to the model and reasoning effort a decision model picks: TypeSafe's Jev (direct or on OpenRouter), OpenRouter's Decisions API, your own Laya server, or vLLM Semantic Router's d3-ultra.
version: 0.2.1
license: MIT
required_open_webui_version: 0.5.0
"""

# A port of jev-claude-router's routing question to Open WebUI. The decision
# model is asked the same two questions (the tier as a choice, the effort as a
# score) with the same criteria, so the confidences mean what they mean there.
# What a tier runs on is yours to name: Open WebUI's model ids are whatever
# your connections call them (`gpt-5-mini`, `anthropic/claude-opus-5-5`,
# `llama3.3:70b`), so each tier has a valve for it.
#
# Everything above the Pipe class is plain Python with no Open WebUI import,
# so the tests run without Open WebUI installed.

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

TIERS = ("haiku", "sonnet", "opus", "fable")

# The efforts Jev's score maps to, low to high: one per effort criterion.
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# What the decision model is told each tier is for. The same lines as
# jev-claude-router's policy.ts: change them there and here together, or the
# confidence bars measured there stop meaning the same thing here.
TIER_CRITERIA = {
    "haiku": (
        "Trivial. A lookup, a rename, a yes or no question, reading one short file, "
        "restating something already on screen."
    ),
    "sonnet": (
        "Straightforward and minor. A small edit whose shape is already obvious from "
        "the request, with no real decision to make."
    ),
    "opus": (
        "Plain implementation carrying some complexity. Writing or changing real code, "
        "possibly across a few files, where the approach is known but the work is not "
        "mechanical."
    ),
    "fable": (
        "High complexity needing higher-order reasoning. Planning, brainstorming, "
        "architecture, systematic debugging, weighing trade-offs, research. Anything "
        "where working out the approach is itself the hard part."
    ),
}

EFFORT_CRITERIA = (
    "The answer is already known or on screen: a lookup, a rename, a yes or "
    "no, restating something.",
    "One or two obvious steps: a small edit whose shape the request already "
    "gives, a short explanation.",
    "Several steps that have to fit together, or a choice worth weighing: "
    "real code across a file or two, a bug with a likely cause.",
    "Many interacting parts, or a subtle failure to chase down: a change "
    "across several files, a bug with no obvious cause, a design with "
    "trade-offs.",
    "Open-ended or ambiguous, or the cost of being wrong is high: "
    "architecture, a systematic debugging campaign, a migration plan, "
    "anything where the approach itself is the hard part.",
)

DEFAULT_TIER_QUESTION = "A developer typed this request to a coding agent. Which model tier should answer it?"
EFFORT_QUESTION = "How much thinking does answering this request take?"

# The decision model reads the whole state it is sent, and grows less accurate
# as it fills with text unrelated to the decision; a request is judged on how
# it opens, so a long paste is cut rather than sent whole and refused.
MAX_STATE_CHARS = 12_000

DEFAULT_TIMEOUT_MS = 1500
# Below this a budget is taken for seconds written where milliseconds are read.
MIN_TIMEOUT_MS = 100
MAX_TIMEOUT_MS = 10_000

DEFAULT_STICKY_CONFIDENCE = 0.75

# How many chats' last tier is kept for stickiness.
MAX_CHATS = 1000

TYPESAFE_BASE_DEFAULT = "https://api.typesafe.ai"
TYPESAFE_MODEL_DEFAULT = "jev-latest"
OPENROUTER_JEV = "typesafe/jev-1.14"

# Routes with a fixed address: the key for a route can only go to its host.
FIXED = {
    "openrouter": "https://openrouter.ai/api/v1/systemone",
    "openrouter-decisions": "https://openrouter.ai/api/alpha/decisions",
    "vllm": "https://decision.vllm-sr.ai/v1/systemone",
}

# Names PROVIDERS accepts, with the other spellings each route goes by.
ROUTE_NAMES = {
    "typesafe": "typesafe",
    "jev": "typesafe",
    "direct": "typesafe",
    "openrouter": "openrouter",
    "systemone": "openrouter",
    "openrouter-decisions": "openrouter-decisions",
    "decisions": "openrouter-decisions",
    "laya": "laya",
    "vllm": "vllm",
    "vllm-sr": "vllm",
    "d3-ultra": "vllm",
    "d3-turbo": "vllm",
    "semantic": "vllm",
}

KEY_NAME = {
    "typesafe": "TYPESAFE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openrouter-decisions": "OPENROUTER_API_KEY",
    "laya": "LAYA_API_KEY",
    "vllm": "VLLM_SR_API_KEY",
}

MODEL_ID = re.compile(r"^[\w.:/@+-]{1,100}$")
HEADER_SAFE = re.compile(r"^[\x21-\x7e]+$")
PLAIN_DECIMAL = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass(frozen=True)
class Route:
    name: str
    endpoint: str
    model: str
    # Empty for a Laya server that asks for none.
    api_key: str


@dataclass
class Decision:
    tier: str
    effort: str
    confidence: float
    effort_confidence: float = 0.0
    probabilities: Optional[dict] = None
    # Who answered: Jev, Laya, or another decision model's id.
    decider: str = "Jev"


@dataclass
class Plan:
    """Where a turn goes and why, for the forward and the status line."""

    tier: str
    model: str
    # Jev's effort, or None when no decision model answered.
    effort: Optional[str]
    status: str
    decision: Optional[Decision] = None
    held: Optional[str] = None
    failed: Optional[str] = None
    ms: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- settings


def flag_on(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def confidence_share_of(raw: Any) -> Optional[float]:
    """
    A confidence bar as a share strictly between 0 and 1, or None. `0.6`, `60`
    and `60%` are the same bar. Without `%`, a number between 1 and 10 must be
    whole: `1.5` is refused rather than read as a bar of 1.5%.
    """
    trimmed = str(raw if raw is not None else "").strip()
    percent = trimmed.endswith("%")
    v = trimmed[:-1].strip() if percent else trimmed
    if not PLAIN_DECIMAL.match(v):
        return None
    n = float(v)
    if not percent and 1 < n < 10 and not n.is_integer():
        return None
    ratio = n / 100 if percent or n > 1 else n
    return ratio if 0 < ratio < 1 else None


def bars_of(raw: Any) -> dict:
    """`microsoft-decision-1=0.85,laya=60%`. A pair that does not parse is left out."""
    out: dict = {}
    for pair in re.split(r"[,;\s]+", str(raw or "")):
        key, sep, value = pair.partition("=")
        key = key.strip().lower()
        bar = confidence_share_of(value)
        if sep and key and re.match(r"^[\w.:@/-]{1,100}$", key) and bar is not None:
            out[key] = bar
    return out


def bar_for(bars: dict, route: Route) -> Optional[float]:
    """The bar for the decider that answered: by full model id, the id after its `/`, or the route name."""
    model = route.model.lower()
    for key in (model, model[model.rfind("/") + 1 :], route.name):
        if key in bars:
            return bars[key]
    return None


def timeout_of(raw: Any) -> int:
    try:
        n = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_MS
    if n < MIN_TIMEOUT_MS:
        return DEFAULT_TIMEOUT_MS
    return min(n, MAX_TIMEOUT_MS)


def effort_map_of(raw: Any) -> dict:
    """
    `low=low,max=high`: what each of Jev's efforts is sent as `reasoning_effort`.
    A level left out goes as its own name; one mapped to nothing (`max=`, `max=none`)
    sends no `reasoning_effort` at all, and the model's own setting applies.
    """
    out = {e: e for e in EFFORTS}
    for pair in re.split(r"[,;\s]+", str(raw or "")):
        key, sep, value = pair.partition("=")
        key = key.strip().lower()
        if not sep or key not in out:
            continue
        value = value.strip()
        out[key] = None if value.lower() in ("", "none", "off", "-") else value
    return out


def tier_list_of(raw: Any) -> set:
    return {t for t in re.split(r"[,;\s]+", str(raw or "").lower()) if t in TIERS}


# ------------------------------------------------------------------ routes


def base_without_endpoint(raw: str) -> str:
    """A base URL without trailing slashes, `/v1` or `/v1/systemone`: copied from the docs' full URL, it would double."""
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    path = re.sub(r"/v1(?:/systemone)?$", "", path)
    return f"{parts.scheme}://{parts.netloc}{path}"


def typesafe_base_of(raw: str, allow_custom: bool) -> tuple[Optional[str], Optional[str]]:
    """
    TypeSafe's base URL, or why it cannot be used. Only api.typesafe.ai and
    hosts under typesafe.ai unless custom bases are allowed: a mistyped valve
    would otherwise send the key elsewhere.
    """
    trimmed = (raw or "").strip()
    if not trimmed:
        return TYPESAFE_BASE_DEFAULT, None
    parts = urlsplit(trimmed)
    if parts.scheme != "https" or not parts.hostname:
        return None, "TYPESAFE_BASE_URL must be an https URL"
    host = parts.hostname.lower()
    if not (host == "api.typesafe.ai" or host.endswith(".typesafe.ai")) and not allow_custom:
        return None, "TYPESAFE_BASE_URL host is not a typesafe.ai host; turn on ALLOW_CUSTOM_BASE to permit it"
    return base_without_endpoint(trimmed), None


def laya_base_of(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Your own Laya server: https, any host, no login in the URL (LAYA_API_KEY goes only there)."""
    trimmed = (raw or "").strip()
    if not trimmed:
        return None, "laya is listed but LAYA_BASE_URL is not set"
    parts = urlsplit(trimmed)
    if parts.scheme != "https" or not parts.hostname:
        return None, "LAYA_BASE_URL must be an https URL"
    if parts.username or parts.password:
        return None, "LAYA_BASE_URL must not carry a login; use LAYA_API_KEY"
    return base_without_endpoint(trimmed), None


def _build(name: str, cfg: dict) -> tuple[Optional[Route], Optional[str]]:
    get = lambda k: str(cfg.get(k) or "").strip()  # noqa: E731
    if name == "typesafe":
        key = get("TYPESAFE_API_KEY")
        if not key:
            return None, "typesafe is listed but TYPESAFE_API_KEY is not set"
        base, why = typesafe_base_of(get("TYPESAFE_BASE_URL"), flag_on(cfg.get("ALLOW_CUSTOM_BASE")))
        if why:
            return None, why
        return Route("typesafe", f"{base}/v1/systemone", get("JEV_MODEL") or TYPESAFE_MODEL_DEFAULT, key), None
    if name == "laya":
        base, why = laya_base_of(get("LAYA_BASE_URL"))
        if why:
            return None, why
        return Route("laya", f"{base}/v1/systemone", get("LAYA_MODEL") or "laya", get("LAYA_API_KEY")), None
    if name in ("openrouter", "openrouter-decisions"):
        key = get("OPENROUTER_API_KEY")
        if not key:
            return None, f"{name} is listed but OPENROUTER_API_KEY is not set"
        model = get("OPENROUTER_DECISIONS_MODEL" if name == "openrouter-decisions" else "OPENROUTER_MODEL")
        return Route(name, FIXED[name], model or OPENROUTER_JEV, key), None
    if name == "vllm":
        key = get("VLLM_SR_API_KEY")
        if not key:
            return None, "vllm is listed but VLLM_SR_API_KEY is not set"
        return Route("vllm", FIXED["vllm"], get("VLLM_SR_MODEL") or "vllm-sr/d3-ultra", key), None
    return None, f"unknown route {name}"


def _checked(route: Route) -> Optional[str]:
    """Why a built route cannot be used, or None. A key with a space or line break cannot go in a header."""
    keyless_laya = route.name == "laya" and route.api_key == ""
    if not keyless_laya and not HEADER_SAFE.match(route.api_key):
        return f"{KEY_NAME[route.name]} has a space, line break or other character a key cannot have"
    if not MODEL_ID.match(route.model):
        return f"the model for {route.name} ({route.model[:40]}) is not a model id"
    return None


def routes_of(cfg: dict) -> tuple[list, Optional[str]]:
    """
    Every route to try, in order: the ones PROVIDERS names, or the one the
    keys choose. OpenRouter is never chosen by its key alone: that key is used
    for other things, and picking it would send every prompt there.
    """
    named = [ROUTE_NAMES.get(n) for n in re.split(r"[\s,;]+", str(cfg.get("PROVIDERS") or "").lower()) if n]
    wanted = list(dict.fromkeys(n for n in named if n))
    if not wanted:
        if str(cfg.get("VLLM_SR_API_KEY") or "").strip():
            wanted = ["vllm"]
        elif str(cfg.get("TYPESAFE_API_KEY") or "").strip():
            wanted = ["typesafe"]
        elif str(cfg.get("LAYA_BASE_URL") or "").strip():
            wanted = ["laya"]
        else:
            return [], "no decision model set up: fill in PROVIDERS and its key"
    routes = []
    for name in wanted:
        route, why = _build(name, cfg)
        if route is None:
            return [], why
        why = _checked(route)
        if why:
            return [], why
        routes.append(route)
    return routes, None


def is_jev(model: str) -> bool:
    return bool(re.search(r"\bjev\b|jev-", model, re.I))


def label_of(route: Route) -> str:
    """What the answerer is called on the status line."""
    if route.name == "laya":
        return "Laya"
    if route.name == "typesafe" or is_jev(route.model):
        return "Jev"
    return route.model[route.model.rfind("/") + 1 :][:24] or "model"


def confidence_source_of(route: Route) -> str:
    """
    Which confidence an answer is read for. `jev`: its `confidence` is Jev's.
    `x_jev`: Laya's own `confidence` means something else, so its
    `x_jev_confidence` is read. `probabilities`: another decision model, whose
    `confidence` has no known meaning; only its probabilities count.
    """
    if route.name == "laya":
        return "x_jev"
    return "jev" if route.name == "typesafe" or is_jev(route.model) else "probabilities"


# ---------------------------------------------------------------- the ask


def head_of(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n]


def state_of(text: str) -> str:
    trimmed = text.strip()
    return trimmed if len(trimmed) <= MAX_STATE_CHARS else f"{head_of(trimmed, MAX_STATE_CHARS)}…"


def request_body_of(state: str, offered: list, model: str, question: str = DEFAULT_TIER_QUESTION) -> dict:
    """Two questions answered in parallel: the tier as a choice, the effort as a score."""
    return {
        "model": model,
        "state": state,
        "questions": {
            "tier": {
                "type": "choice",
                "instructions": question or DEFAULT_TIER_QUESTION,
                "criteria": {t: TIER_CRITERIA[t] for t in offered},
            },
            "effort": {
                "type": "score",
                "instructions": EFFORT_QUESTION,
                "criteria": list(EFFORT_CRITERIA),
            },
        },
    }


def without_secrets(text: str, key: str) -> str:
    """An error as a few safe words: the key and any Bearer credential taken out, one line, short."""
    if key and len(key) >= 4:
        text = text.replace(key, "…")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer …", text)
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())
    return text if len(text) <= 80 else f"{text[:77]}…"


def provider_note_of(text: str) -> str:
    """The provider's own error type, when it sent one."""
    try:
        kind = json.loads(text).get("error", {}).get("type")
    except (ValueError, AttributeError):
        return ""
    if not isinstance(kind, str):
        return ""
    plain = re.sub(r"[^\w.-]", "", kind)[:40]
    return f" ({plain})" if plain else ""


# fetch(url, headers, body, timeout_seconds) -> (status, text)
Fetch = Callable[[str, dict, str, float], Awaitable[tuple]]


async def aiohttp_fetch(url: str, headers: dict, body: str, timeout_s: float) -> tuple:
    import aiohttp  # Open WebUI ships it.

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(url, headers=headers, data=body.encode("utf-8")) as response:
            return response.status, await response.text()


async def ask(route: Route, state: str, offered: list, timeout_s: float, fetch: Fetch, question: str = "") -> dict:
    """One attempt at one route: `{"ok": True, "answers": ...}` or `{"ok": False, "reason": ...}`."""
    headers = {"content-type": "application/json"}
    if route.api_key:
        headers["authorization"] = f"Bearer {route.api_key}"
    body = json.dumps(request_body_of(state_of(state), offered, route.model, question))
    try:
        status, text = await asyncio.wait_for(fetch(route.endpoint, headers, body, timeout_s), timeout_s)
    except asyncio.TimeoutError:
        return {"ok": False, "reason": f"timed out after {round(timeout_s * 1000)}ms", "timeout": True}
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - any failure is a named pass for the turn
        return {"ok": False, "reason": f"request failed: {without_secrets(str(error) or type(error).__name__, route.api_key)}"}
    if not 200 <= int(status) < 300:
        return {"ok": False, "reason": f"{route.name} said HTTP {status}{provider_note_of(text)}"}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"ok": False, "reason": "response was not JSON"}
    if not isinstance(parsed, dict) or not parsed.get("answers"):
        return {"ok": False, "reason": "response carried no answers"}
    return {"ok": True, "answers": parsed["answers"], "route": route}


async def ask_chain(
    routes: list, state: str, offered: list, timeout_ms: int, fetch: Fetch, question: str = "", now=time.monotonic
) -> dict:
    """
    Asks each route in turn until one answers, within one shared budget: a
    route that fails at once leaves the rest of the time to the next; one that
    times out has spent it.
    """
    started = now()
    failures = []
    for route in routes:
        left_ms = timeout_ms - (now() - started) * 1000
        if left_ms < MIN_TIMEOUT_MS:
            break
        result = await ask(route, state, offered, left_ms / 1000, fetch, question)
        result["ms"] = round((now() - started) * 1000)
        if result["ok"]:
            return result
        failures.append((route.name, result["reason"]))
        if result.get("timeout"):
            break
    if not failures:
        reason = "no route had time left"
    elif len(routes) == 1:
        reason = failures[0][1]
    else:
        reason = "; then ".join(r if r.startswith(n) else f"{n}: {r}" for n, r in failures)
    return {"ok": False, "reason": reason, "ms": round((now() - started) * 1000)}


# ------------------------------------------------------------ the answer


def effort_of(score: Any) -> str:
    """The score across EFFORT_CRITERIA to the nearest effort."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score != score or score in (float("inf"), float("-inf")):
        return "medium"
    # Half rounds up, as jev-claude-router's Math.round does: Python's round(2.5) is 2.
    return EFFORTS[min(max(math.floor(score + 0.5), 0), len(EFFORTS) - 1)]


def _number(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return None
    return min(1.0, max(0.0, float(v)))


def probabilities_of(raw: Any, offered: list) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    out = {t: _number(raw.get(t)) for t in offered}
    out = {t: p for t, p in out.items() if p is not None}
    return out or None


def confidence_from(probabilities: Optional[dict], options: int, chosen: Optional[str] = None) -> float:
    """
    TypeSafe's confidence from the probabilities, `(n·p − 1) / (n − 1)`: all
    the mass on one option gives 1, an even spread 0. With `chosen`, the mass
    is the chosen tier's own, so an answer whose choice and probabilities
    disagree does not clear a bar as if it were sure.
    """
    if not probabilities:
        return 0.0
    p = probabilities.get(chosen, 0.0) if chosen is not None else max(probabilities.values())
    if options <= 1:
        return min(1.0, max(0.0, p))
    return min(1.0, max(0.0, (options * p - 1) / (options - 1)))


def spread_confidence(raw: Any) -> float:
    """A score answer's confidence from its probabilities, an object or a list over the levels."""
    values = list(raw.values()) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    values = [v for v in (_number(x) for x in values) if v is not None]
    if len(values) < 2:
        return 0.0
    n = len(values)
    return min(1.0, max(0.0, (n * max(values) - 1) / (n - 1)))


def decision_of(answers: Any, offered: list, source: str = "jev", decider: str = "Jev") -> Optional[Decision]:
    """
    An `answers` object as a decision, or None when it is missing, malformed,
    or names a tier that was not offered.
    """
    if not isinstance(answers, dict):
        return None
    tier = answers.get("tier")
    if not isinstance(tier, dict) or tier.get("type") != "choice":
        return None
    choice = tier.get("choice")
    if not isinstance(choice, str) or choice not in offered:
        return None

    def read(answer: dict) -> Optional[float]:
        if source == "probabilities":
            return None
        return _number(answer.get("x_jev_confidence" if source == "x_jev" else "confidence"))

    probabilities = probabilities_of(tier.get("probabilities"), offered)
    confidence = read(tier)
    if confidence is None:
        confidence = confidence_from(probabilities, len(offered), choice)
    effort = answers.get("effort")
    effort_confidence = 0.0
    if isinstance(effort, dict):
        effort_confidence = read(effort)
        if effort_confidence is None:
            effort_confidence = spread_confidence(effort.get("probabilities")) if source == "probabilities" else 0.0
    return Decision(
        tier=choice,
        effort=effort_of(effort.get("score") if isinstance(effort, dict) else None),
        confidence=confidence,
        effort_confidence=effort_confidence,
        probabilities=probabilities,
        decider=decider,
    )


def sticky_tier(fresh: Decision, previous: Optional[str], bar: float, offered: list) -> tuple[str, Optional[str]]:
    """
    The tier to run on and the tier held from, if any. A switch away from the
    tier this chat last ran on has to clear the bar; a shaky one stays, which
    keeps a provider's prompt cache warm. The effort is applied either way.
    """
    if previous is None or previous == fresh.tier or previous not in offered:
        return fresh.tier, None
    if fresh.confidence < bar:
        return previous, fresh.tier
    return fresh.tier, None


def nearest_tier(wanted: str, offered: list) -> str:
    """`wanted` if it is offered, else the nearest offered tier above it, else below."""
    if wanted in offered:
        return wanted
    at = TIERS.index(wanted) if wanted in TIERS else 1
    for t in TIERS[at + 1 :]:
        if t in offered:
            return t
    for t in reversed(TIERS[:at]):
        if t in offered:
            return t
    return offered[0]


def text_of(content: Any) -> str:
    """A message's text: a string, or the text parts of a list of parts (images left out)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    return ""


def last_user_text(messages: Any) -> str:
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "user":
            return text_of(message.get("content"))
    return ""


EFFORT_SLOT = "{effort}"


def resolved_model(template: str, effort: Optional[str], default: str = "medium") -> str:
    """
    A model id with `{effort}` in it carries the effort in its name
    (`cursor/claude-opus-5-5-{effort}` → `cursor/claude-opus-5-5-high`), for
    connections that list one model per effort. With no effort known, `default`.
    """
    return template.replace(EFFORT_SLOT, effort or default)


def pct(x: float) -> str:
    return f"{round(x * 100)}%"


async def _maybe_await(value: Any) -> Any:
    """Open WebUI 0.9 made its model methods async; before that they were not."""
    return await value if inspect.isawaitable(value) else value


# -------------------------------------------------------------------- pipe


class Pipe:
    class Valves(BaseModel):
        # Where each tier runs. A tier left empty is not offered to the decision model.
        HAIKU_MODEL: str = Field(
            default="",
            description="Open WebUI model id for trivial requests (Jev's 'haiku' tier): a lookup, a yes or no, restating. E.g. gpt-5-mini, anthropic/claude-haiku-5-5, llama3.2:3b. {effort} in the id is replaced by the effort (cursor/claude-haiku-5-5-{effort}), and then no reasoning_effort is sent.",
        )
        SONNET_MODEL: str = Field(
            default="",
            description="Model id for straightforward, minor requests (Jev's 'sonnet' tier).",
        )
        OPUS_MODEL: str = Field(
            default="",
            description="Model id for real work with some complexity (Jev's 'opus' tier).",
        )
        FABLE_MODEL: str = Field(
            default="",
            description="Model id for planning, architecture, research, hard debugging (Jev's 'fable' tier).",
        )

        PROVIDERS: str = Field(
            default="",
            description="Decision models to ask, in order, comma-separated: typesafe (Jev direct), openrouter (Jev on OpenRouter, System One), openrouter-decisions (OpenRouter's Decisions API), laya (your own server), vllm (d3-ultra). E.g. 'vllm,openrouter'. Empty: chosen by the keys set (vllm, then typesafe, then laya; OpenRouter only by name).",
        )
        TIMEOUT_MS: int = Field(
            default=DEFAULT_TIMEOUT_MS,
            description="How long a turn waits for the decision, all routes together, in milliseconds (100 to 10000).",
        )

        TYPESAFE_API_KEY: str = Field(default="", description="TypeSafe key, for typesafe.", json_schema_extra={"input": {"type": "password"}})
        TYPESAFE_BASE_URL: str = Field(default="", description="TypeSafe API base; https://api.typesafe.ai when empty. Only typesafe.ai hosts unless ALLOW_CUSTOM_BASE.")
        ALLOW_CUSTOM_BASE: bool = Field(default=False, description="Let TYPESAFE_BASE_URL point at a host outside typesafe.ai.")
        JEV_MODEL: str = Field(default="", description="Jev model on typesafe; jev-latest when empty. TypeSafe advises pinning one (jev-1.13.0).")

        OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter key, for openrouter and openrouter-decisions.", json_schema_extra={"input": {"type": "password"}})
        OPENROUTER_MODEL: str = Field(default="", description=f"Model on openrouter (System One); {OPENROUTER_JEV} when empty.")
        OPENROUTER_DECISIONS_MODEL: str = Field(
            default="",
            description=f"Model on openrouter-decisions; {OPENROUTER_JEV} when empty. Any decision model in OpenRouter's catalog (perplexity/pplx-decider-v1.2, microsoft/microsoft-decision-1); one that is not Jev is read for its probabilities only.",
        )

        LAYA_BASE_URL: str = Field(default="", description="Your Laya server's https address, for laya.")
        LAYA_API_KEY: str = Field(default="", description="Laya key, if your server asks for one.", json_schema_extra={"input": {"type": "password"}})
        LAYA_MODEL: str = Field(default="", description="Model on your Laya server; laya when empty.")

        VLLM_SR_API_KEY: str = Field(default="", description="vLLM Semantic Router key, for vllm (d3-ultra).", json_schema_extra={"input": {"type": "password"}})
        VLLM_SR_MODEL: str = Field(default="", description="Model on vllm; vllm-sr/d3-ultra when empty.")

        STICKY: bool = Field(default=True, description="Hold a chat on the tier it last ran on when the decision model is not sure of a switch.")
        STICKY_CONFIDENCE: str = Field(default="0.75", description="The confidence a switch must clear: 0.75, 75 or 75%.")
        DECIDER_BARS: str = Field(
            default="perplexity/pplx-decider-v1.2=0.5",
            description="A bar per decision model whose confidence is not Jev's: model id, id after the /, or route name. E.g. 'd3-ultra=0.6,laya=60%'.",
        )
        FALLBACK_TIER: Literal["haiku", "sonnet", "opus", "fable"] = Field(
            default="sonnet",
            description="Where a chat's first turn goes when no decision model answers (later turns stay where they were). The nearest filled-in tier if this one is empty.",
        )
        TASK_TIER: Literal["lowest", "haiku", "sonnet", "opus", "fable"] = Field(
            default="lowest",
            description="Where Open WebUI's background tasks (titles, tags, follow-ups, queries) go, without asking the decision model.",
        )

        SEND_REASONING_EFFORT: bool = Field(default=True, description="Send the decision model's effort as reasoning_effort. A reasoning_effort already in the request wins.")
        EFFORT_TIERS: str = Field(
            default="haiku,sonnet,opus,fable",
            description="Tiers whose model takes reasoning_effort. Leave a tier out when its model refuses the parameter.",
        )
        EFFORT_MAP: str = Field(
            default="low=low,medium=medium,high=high,xhigh=high,max=high",
            description="What each effort is sent as. Most OpenAI-compatible models take low, medium and high; map a level to nothing (max=) to send none.",
        )
        TIER_QUESTION: str = Field(
            default=DEFAULT_TIER_QUESTION,
            description="The question the tier is asked with. Changing it moves the confidences the bars are tuned against.",
        )
        SHOW_STATUS: bool = Field(default=True, description="Show where each turn went above the reply.")

    def __init__(self):
        self.valves = self.Valves()
        self._last_tier: OrderedDict = OrderedDict()
        # Swapped in tests.
        self._fetch: Fetch = aiohttp_fetch

    # Keys and addresses can come from the environment instead (a Docker
    # deployment's secrets); a valve that is filled in wins.
    _ENV_FALLBACK = (
        "PROVIDERS",
        "TYPESAFE_API_KEY",
        "TYPESAFE_BASE_URL",
        "JEV_MODEL",
        "OPENROUTER_API_KEY",
        "LAYA_BASE_URL",
        "LAYA_API_KEY",
        "LAYA_MODEL",
        "VLLM_SR_API_KEY",
        "VLLM_SR_MODEL",
    )

    def config(self) -> dict:
        cfg = self.valves.model_dump() if hasattr(self.valves, "model_dump") else self.valves.dict()
        for name in self._ENV_FALLBACK:
            if not str(cfg.get(name) or "").strip():
                env = "JEV_ROUTER_PROVIDER" if name == "PROVIDERS" else "JEV_ROUTER_JEV_MODEL" if name == "JEV_MODEL" else name
                cfg[name] = os.environ.get(env, "")
        return cfg

    def tier_models(self) -> dict:
        v = self.valves
        models = {"haiku": v.HAIKU_MODEL, "sonnet": v.SONNET_MODEL, "opus": v.OPUS_MODEL, "fable": v.FABLE_MODEL}
        return {t: m.strip() for t, m in models.items() if m and m.strip()}

    def remember(self, chat_id: Optional[str], tier: str) -> None:
        if not chat_id:
            return
        self._last_tier[chat_id] = tier
        self._last_tier.move_to_end(chat_id)
        while len(self._last_tier) > MAX_CHATS:
            self._last_tier.popitem(last=False)

    async def decide(self, messages: Any, chat_id: Optional[str]) -> Plan:
        """Where this turn goes: the decision model's pick, held or fallen back as the valves say."""
        models = self.tier_models()
        offered = [t for t in TIERS if t in models]
        previous = self._last_tier.get(chat_id) if chat_id else None
        if previous not in offered:
            previous = None

        def fallback(why: str, ms: int = 0) -> Plan:
            tier = previous or nearest_tier(self.valves.FALLBACK_TIER, offered)
            where = "stayed" if previous else "fallback"
            model = resolved_model(models[tier], None)
            return Plan(tier, model, None, f"{tier} · {where}: {why} → {model}", failed=why, ms=ms)

        if len(offered) == 1:
            only = offered[0]
            model = resolved_model(models[only], None)
            return Plan(only, model, None, f"{only} · the only tier filled in → {model}")

        cfg = self.config()
        routes, why = routes_of(cfg)
        if why:
            return fallback(why)
        state = last_user_text(messages)
        if not state.strip():
            return fallback("no text to judge")

        result = await ask_chain(routes, state, offered, timeout_of(cfg.get("TIMEOUT_MS")), self._fetch, cfg.get("TIER_QUESTION") or "")
        if not result["ok"]:
            return fallback(f"no decision ({result['reason']})", result["ms"])
        route: Route = result["route"]
        fresh = decision_of(result["answers"], offered, confidence_source_of(route), label_of(route))
        if fresh is None:
            return fallback(f"{label_of(route)} gave no usable answer", result["ms"])

        bar = bar_for(bars_of(cfg.get("DECIDER_BARS")), route)
        if bar is None:
            bar = confidence_share_of(cfg.get("STICKY_CONFIDENCE")) or DEFAULT_STICKY_CONFIDENCE
        tier, held = (fresh.tier, None)
        if flag_on(cfg.get("STICKY")):
            tier, held = sticky_tier(fresh, previous, bar, offered)

        said = f"{fresh.decider} {pct(fresh.confidence)}"
        model = resolved_model(models[tier], fresh.effort)
        if held:
            status = f"{tier} · {fresh.effort} · held: {fresh.decider} said {held} at {pct(fresh.confidence)}, bar {pct(bar)} · {result['ms']}ms → {model}"
        else:
            status = f"{tier} · {fresh.effort} · {said} · {result['ms']}ms → {model}"
        return Plan(tier, model, fresh.effort, status, decision=fresh, held=held, ms=result["ms"], extra={"template": models[tier]})

    def task_plan(self) -> Optional[Plan]:
        models = self.tier_models()
        offered = [t for t in TIERS if t in models]
        if not offered:
            return None
        wanted = self.valves.TASK_TIER
        tier = offered[0] if wanted == "lowest" else nearest_tier(wanted, offered)
        model = resolved_model(models[tier], None, "low")
        return Plan(tier, model, None, f"task → {model}", extra={"template": models[tier]})

    def forward_body(self, body: dict, plan: Plan) -> dict:
        out = {**body, "model": plan.model}
        v = self.valves
        # An effort already in the model's name is not sent a second time.
        if (
            plan.effort is not None
            and EFFORT_SLOT not in plan.extra.get("template", "")
            and v.SEND_REASONING_EFFORT
            and plan.tier in tier_list_of(v.EFFORT_TIERS)
            and out.get("reasoning_effort") in (None, "")
        ):
            sent = effort_map_of(v.EFFORT_MAP).get(plan.effort)
            if sent:
                out["reasoning_effort"] = sent
        return out

    def loops_back(self, own: str) -> Optional[str]:
        """A tier model that is this pipe itself would call it again on every turn."""
        prefix = own.split(".", 1)[0]
        for tier, model in self.tier_models().items():
            if own and (model == own or model.split(".", 1)[0] == prefix):
                return f"{tier.upper()}_MODEL is this router itself ({model}); name the model it should hand the turn to"
        return None

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Any = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[Any]]] = None,
        __task__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> Any:
        if not self.tier_models():
            return "Jev Router has no models to route to yet: fill in HAIKU_MODEL, SONNET_MODEL, OPUS_MODEL and/or FABLE_MODEL in its valves (Admin Panel → Functions → Jev Router → ⚙)."
        loop = self.loops_back(str(body.get("model") or ""))
        if loop:
            return f"Jev Router: {loop}."

        metadata = __metadata__ or body.get("metadata") or {}
        chat_id = metadata.get("chat_id") if isinstance(metadata, dict) else None
        task = __task__ or (metadata.get("task") if isinstance(metadata, dict) else None)

        if task:
            plan = self.task_plan()
        else:
            plan = await self.decide(body.get("messages"), chat_id)
            self.remember(chat_id, plan.tier)
            if self.valves.SHOW_STATUS and __event_emitter__ is not None:
                try:
                    await __event_emitter__({"type": "status", "data": {"description": plan.status, "done": True}})
                except Exception:  # noqa: BLE001 - a status that cannot be shown must not cost the reply
                    pass

        from open_webui.models.users import Users
        from open_webui.utils.chat import generate_chat_completion

        user = await _maybe_await(Users.get_user_by_id(__user__["id"])) if __user__ else None
        return await generate_chat_completion(__request__, self.forward_body(body, plan), user)
