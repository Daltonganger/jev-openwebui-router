import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jev_router as jr  # noqa: E402

FOUR = ["haiku", "sonnet", "opus", "fable"]


def run(coro):
    return asyncio.run(coro)


def jev_answer(choice="opus", confidence=0.9, score=2, probabilities=None, **extra):
    tier = {"type": "choice", "choice": choice, "confidence": confidence}
    if probabilities is not None:
        tier["probabilities"] = probabilities
    tier.update(extra)
    return json.dumps({"answers": {"tier": tier, "effort": {"type": "score", "score": score, "confidence": 0.7}}})


class FakeFetch:
    """Answers each call from a list: (status, text), or an exception to raise, or 'hang'."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, url, headers, body, timeout_s):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body), "timeout_s": timeout_s})
        reply = self.replies.pop(0) if self.replies else (200, jev_answer())
        if reply == "hang":
            await asyncio.sleep(10)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_pipe(**valves):
    pipe = jr.Pipe()
    defaults = dict(HAIKU_MODEL="small", SONNET_MODEL="medium", OPUS_MODEL="large", FABLE_MODEL="huge", TYPESAFE_API_KEY="ts-key-123")
    defaults.update(valves)
    pipe.valves = pipe.Valves(**defaults)
    return pipe


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for name in ("JEV_ROUTER_PROVIDER", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "LAYA_BASE_URL", "LAYA_API_KEY", "VLLM_SR_API_KEY", "JEV_ROUTER_JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------- routes


def test_named_routes_in_order_with_their_fixed_hosts():
    routes, why = jr.routes_of(
        {"PROVIDERS": "vllm, openrouter-decisions;laya openrouter", "VLLM_SR_API_KEY": "v-key", "OPENROUTER_API_KEY": "or-key", "LAYA_BASE_URL": "https://laya.example.com/v1/systemone/"}
    )
    assert why is None
    assert [(r.name, r.endpoint, r.model) for r in routes] == [
        ("vllm", "https://decision.vllm-sr.ai/v1/systemone", "vllm-sr/d3-turbo"),
        ("openrouter-decisions", "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.14"),
        ("laya", "https://laya.example.com/v1/systemone", "laya"),
        ("openrouter", "https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.14"),
    ]
    assert routes[2].api_key == ""


def test_aliases_and_duplicates():
    routes, _ = jr.routes_of({"PROVIDERS": "d3-turbo,jev,vllm-sr,decisions", "VLLM_SR_API_KEY": "v", "TYPESAFE_API_KEY": "t", "OPENROUTER_API_KEY": "o"})
    assert [r.name for r in routes] == ["vllm", "typesafe", "openrouter-decisions"]


def test_each_openrouter_route_has_its_own_model():
    routes, _ = jr.routes_of(
        {"PROVIDERS": "openrouter,openrouter-decisions", "OPENROUTER_API_KEY": "o", "OPENROUTER_MODEL": "typesafe/jev-1.13", "OPENROUTER_DECISIONS_MODEL": "perplexity/pplx-decider-v1.2"}
    )
    assert [r.model for r in routes] == ["typesafe/jev-1.13", "perplexity/pplx-decider-v1.2"]


def test_chosen_by_keys_never_openrouter_alone():
    assert jr.routes_of({"OPENROUTER_API_KEY": "o"})[0] == []
    assert jr.routes_of({"TYPESAFE_API_KEY": "t"})[0][0].name == "typesafe"
    assert jr.routes_of({"TYPESAFE_API_KEY": "t", "VLLM_SR_API_KEY": "v"})[0][0].name == "vllm"
    assert jr.routes_of({"LAYA_BASE_URL": "https://laya.example.com"})[0][0].name == "laya"


def test_missing_key_for_a_named_route_is_an_error():
    routes, why = jr.routes_of({"PROVIDERS": "vllm,openrouter", "VLLM_SR_API_KEY": "v"})
    assert routes == [] and "OPENROUTER_API_KEY" in why


def test_typesafe_base_is_allowlisted():
    assert jr.routes_of({"TYPESAFE_API_KEY": "t", "TYPESAFE_BASE_URL": "https://eu.typesafe.ai/v1"})[0][0].endpoint == "https://eu.typesafe.ai/v1/systemone"
    assert "not a typesafe.ai host" in jr.routes_of({"TYPESAFE_API_KEY": "t", "TYPESAFE_BASE_URL": "https://evil.example.com"})[1]
    assert "https" in jr.routes_of({"TYPESAFE_API_KEY": "t", "TYPESAFE_BASE_URL": "http://api.typesafe.ai"})[1]
    allowed = jr.routes_of({"TYPESAFE_API_KEY": "t", "TYPESAFE_BASE_URL": "https://my.proxy.dev", "ALLOW_CUSTOM_BASE": True})
    assert allowed[0][0].endpoint == "https://my.proxy.dev/v1/systemone"


def test_laya_url_rules():
    assert "login" in jr.routes_of({"PROVIDERS": "laya", "LAYA_BASE_URL": "https://u:p@laya.example.com"})[1]
    assert "https" in jr.routes_of({"PROVIDERS": "laya", "LAYA_BASE_URL": "http://laya.local"})[1]
    assert "not set" in jr.routes_of({"PROVIDERS": "laya"})[1]


def test_keys_and_models_that_cannot_be_sent_are_refused():
    assert "character" in jr.routes_of({"TYPESAFE_API_KEY": "has space"})[1]
    assert "not a model id" in jr.routes_of({"TYPESAFE_API_KEY": "t", "JEV_MODEL": "bad model"})[1]


def test_who_answered_and_which_confidence_counts():
    ts = jr.Route("typesafe", "", "jev-latest", "k")
    laya = jr.Route("laya", "", "multilingual", "")
    vllm = jr.Route("vllm", "", "vllm-sr/d3-turbo", "k")
    or_jev = jr.Route("openrouter-decisions", "", "typesafe/jev-1.14", "k")
    pplx = jr.Route("openrouter-decisions", "", "perplexity/pplx-decider-v1.2", "k")
    assert [jr.label_of(r) for r in (ts, laya, vllm, or_jev, pplx)] == ["Jev", "Laya", "d3-turbo", "Jev", "pplx-decider-v1.2"]
    assert [jr.confidence_source_of(r) for r in (ts, laya, vllm, or_jev, pplx)] == ["jev", "x_jev", "probabilities", "jev", "probabilities"]


# ---------------------------------------------------------------- request


def test_request_body_offers_only_the_tiers_filled_in():
    body = jr.request_body_of("hello", ["haiku", "opus"], "jev-latest")
    assert body["model"] == "jev-latest" and body["state"] == "hello"
    assert list(body["questions"]["tier"]["criteria"]) == ["haiku", "opus"]
    assert body["questions"]["tier"]["criteria"]["opus"] == jr.TIER_CRITERIA["opus"]
    assert len(body["questions"]["effort"]["criteria"]) == 5


def test_long_prompt_is_cut():
    assert len(jr.state_of("x" * 20_000)) == jr.MAX_STATE_CHARS + 1


def test_ask_sends_bearer_and_reads_answers():
    route = jr.Route("typesafe", "https://api.typesafe.ai/v1/systemone", "jev-latest", "ts-key")
    fetch = FakeFetch((200, jev_answer()))
    result = run(jr.ask(route, "write a parser", FOUR, 1.0, fetch))
    assert result["ok"]
    assert fetch.calls[0]["headers"]["authorization"] == "Bearer ts-key"
    assert fetch.calls[0]["url"] == route.endpoint


def test_keyless_laya_sends_no_authorization():
    route = jr.Route("laya", "https://laya.example.com/v1/systemone", "laya", "")
    fetch = FakeFetch((200, jev_answer()))
    run(jr.ask(route, "hi", FOUR, 1.0, fetch))
    assert "authorization" not in fetch.calls[0]["headers"]


def test_errors_are_named_and_keys_scrubbed():
    route = jr.Route("vllm", "u", "vllm-sr/d3-turbo", "secret-key-999")
    r = run(jr.ask(route, "hi", FOUR, 1.0, FakeFetch((401, '{"error":{"type":"auth_error"}}'))))
    assert r == {"ok": False, "reason": "vllm said HTTP 401 (auth_error)"}
    r = run(jr.ask(route, "hi", FOUR, 1.0, FakeFetch(RuntimeError("boom with secret-key-999 and Bearer abc123"))))
    assert "secret-key-999" not in r["reason"] and "abc123" not in r["reason"]
    assert run(jr.ask(route, "hi", FOUR, 1.0, FakeFetch((200, "not json"))))["reason"] == "response was not JSON"
    assert run(jr.ask(route, "hi", FOUR, 1.0, FakeFetch((200, "{}"))))["reason"] == "response carried no answers"


def test_chain_falls_through_to_the_next_route():
    a = jr.Route("vllm", "a", "vllm-sr/d3-turbo", "k")
    b = jr.Route("openrouter", "b", "typesafe/jev-1.14", "k")
    fetch = FakeFetch((503, ""), (200, jev_answer()))
    result = run(jr.ask_chain([a, b], "hi", FOUR, 1500, fetch))
    assert result["ok"] and result["route"] is b
    assert [c["url"] for c in fetch.calls] == ["a", "b"]


def test_chain_stops_after_a_timeout_and_says_who_said_what():
    a = jr.Route("vllm", "a", "vllm-sr/d3-turbo", "k")
    b = jr.Route("openrouter", "b", "typesafe/jev-1.14", "k")
    fetch = FakeFetch("hang", (200, jev_answer()))
    result = run(jr.ask_chain([a, b], "hi", FOUR, 150, fetch))
    assert not result["ok"] and len(fetch.calls) == 1
    assert result["reason"].startswith("vllm: timed out")
    both = run(jr.ask_chain([a, b], "hi", FOUR, 1500, FakeFetch((500, ""), (429, ""))))
    assert both["reason"] == "vllm said HTTP 500; then openrouter said HTTP 429"


# ----------------------------------------------------------------- answer


def test_decision_from_jev_confidence():
    d = jr.decision_of(json.loads(jev_answer("fable", 0.81, 3.6))["answers"], FOUR)
    assert (d.tier, d.effort, d.confidence) == ("fable", "max", 0.81)


def test_effort_rounds_half_up_and_clamps():
    assert [jr.effort_of(s) for s in (0, 0.49, 0.5, 2.5, 9, -3, None, "2", True)] == ["low", "low", "medium", "xhigh", "max", "low", "medium", "medium", "medium"]


def test_tier_not_offered_is_no_decision():
    assert jr.decision_of(json.loads(jev_answer("fable"))["answers"], ["haiku", "sonnet"]) is None
    assert jr.decision_of({"tier": {"type": "score"}}, FOUR) is None
    assert jr.decision_of("nope", FOUR) is None


def test_probabilities_only_for_other_models():
    answers = json.loads(jev_answer("opus", 0.99, 2, {"haiku": 0.05, "sonnet": 0.05, "opus": 0.85, "fable": 0.05}))["answers"]
    d = jr.decision_of(answers, FOUR, "probabilities", "d3-turbo")
    assert d.confidence == pytest.approx((4 * 0.85 - 1) / 3)
    assert d.decider == "d3-turbo"


def test_laya_reads_x_jev_confidence_never_confidence():
    answers = json.loads(jev_answer("opus", 0.2, 2, x_jev_confidence=0.88))["answers"]
    assert jr.decision_of(answers, FOUR, "x_jev").confidence == 0.88
    no_x = json.loads(jev_answer("opus", 0.99, 2, {"opus": 0.25, "haiku": 0.25, "sonnet": 0.25, "fable": 0.25}))["answers"]
    assert jr.decision_of(no_x, FOUR, "x_jev").confidence == 0


def test_choice_and_probabilities_disagreeing_is_not_sure():
    answers = json.loads(jev_answer("haiku", None, 1, {"haiku": 0.05, "fable": 0.95}))["answers"]
    assert jr.decision_of(answers, FOUR).confidence == 0


def test_bars():
    assert [jr.confidence_share_of(x) for x in ("0.6", "60", "60%", "1.5", "0", "100", "abc", "0x40")] == [0.6, 0.6, 0.6, None, None, None, None, None]
    bars = jr.bars_of("perplexity/pplx-decider-v1.2=0.5, d3-turbo=60%;laya=0.7 bad=x")
    assert bars == {"perplexity/pplx-decider-v1.2": 0.5, "d3-turbo": 0.6, "laya": 0.7}
    assert jr.bar_for(bars, jr.Route("vllm", "", "vllm-sr/d3-turbo", "k")) == 0.6
    assert jr.bar_for(bars, jr.Route("laya", "", "multilingual", "")) == 0.7
    assert jr.bar_for(bars, jr.Route("typesafe", "", "jev-latest", "k")) is None


def test_sticky():
    shaky = jr.Decision("opus", "high", 0.6)
    sure = jr.Decision("opus", "high", 0.9)
    assert jr.sticky_tier(shaky, None, 0.75, FOUR) == ("opus", None)
    assert jr.sticky_tier(shaky, "sonnet", 0.75, FOUR) == ("sonnet", "opus")
    assert jr.sticky_tier(sure, "sonnet", 0.75, FOUR) == ("opus", None)
    assert jr.sticky_tier(shaky, "fable", 0.75, ["sonnet", "opus"]) == ("opus", None)


def test_nearest_tier():
    assert jr.nearest_tier("sonnet", ["haiku", "opus"]) == "opus"
    assert jr.nearest_tier("fable", ["haiku", "opus"]) == "opus"
    assert jr.nearest_tier("haiku", ["haiku"]) == "haiku"


def test_user_text_from_parts():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": [{"type": "text", "text": "look at this"}, {"type": "image_url", "image_url": {"url": "data:"}}]},
    ]
    assert jr.last_user_text(messages) == "look at this"


def test_effort_map():
    m = jr.effort_map_of("low=low,medium=medium,high=high,xhigh=high,max=")
    assert m == {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": None}


# ------------------------------------------------------------------- pipe


def test_decide_routes_to_the_model_filled_in():
    pipe = make_pipe()
    pipe._fetch = FakeFetch((200, jev_answer("fable", 0.9, 4)))
    plan = run(pipe.decide([{"role": "user", "content": "design a migration"}], "chat-1"))
    assert (plan.tier, plan.model, plan.effort) == ("fable", "huge", "max")
    assert plan.status.startswith("fable · max · Jev 90% ·") and plan.status.endswith("→ huge")


def test_decide_offers_only_filled_in_tiers():
    pipe = make_pipe(FABLE_MODEL="", HAIKU_MODEL="")
    pipe._fetch = fetch = FakeFetch((200, jev_answer("opus")))
    run(pipe.decide([{"role": "user", "content": "x"}], None))
    assert list(fetch.calls[0]["body"]["questions"]["tier"]["criteria"]) == ["sonnet", "opus"]


def test_decide_holds_a_shaky_switch_in_the_same_chat():
    pipe = make_pipe()
    pipe._fetch = FakeFetch((200, jev_answer("sonnet", 0.9, 1)), (200, jev_answer("opus", 0.6, 3)))
    first = run(pipe.decide([{"role": "user", "content": "a"}], "c"))
    pipe.remember("c", first.tier)
    second = run(pipe.decide([{"role": "user", "content": "b"}], "c"))
    assert (second.tier, second.held, second.effort) == ("sonnet", "opus", "xhigh")
    assert "held: Jev said opus at 60%, bar 75%" in second.status


def test_decide_uses_a_decider_bar():
    pipe = make_pipe(PROVIDERS="vllm", VLLM_SR_API_KEY="v", DECIDER_BARS="d3-turbo=0.5")
    probs = {"haiku": 0.1, "sonnet": 0.1, "opus": 0.7, "fable": 0.1}  # confidence 0.6
    pipe._fetch = FakeFetch((200, jev_answer("opus", None, 2, probs)))
    pipe.remember("c", "sonnet")
    plan = run(pipe.decide([{"role": "user", "content": "b"}], "c"))
    assert plan.tier == "opus" and plan.held is None and "d3-turbo 60%" in plan.status


def test_decide_falls_back_and_stays():
    pipe = make_pipe(FALLBACK_TIER="fable", FABLE_MODEL="")
    pipe._fetch = FakeFetch((500, ""))
    plan = run(pipe.decide([{"role": "user", "content": "x"}], "c"))
    assert plan.tier == "opus" and plan.effort is None and "fallback" in plan.status and "HTTP 500" in plan.status
    pipe.remember("c", "haiku")
    pipe._fetch = FakeFetch((500, ""))
    assert run(pipe.decide([{"role": "user", "content": "x"}], "c")).tier == "haiku"


def test_decide_misconfigured_route_falls_back_with_the_reason():
    pipe = make_pipe(PROVIDERS="openrouter")
    plan = run(pipe.decide([{"role": "user", "content": "x"}], None))
    assert plan.tier == "sonnet" and "OPENROUTER_API_KEY" in plan.status


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("JEV_ROUTER_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_SR_API_KEY", "from-env")
    cfg = make_pipe(TYPESAFE_API_KEY="").config()
    assert cfg["PROVIDERS"] == "vllm" and cfg["VLLM_SR_API_KEY"] == "from-env"
    assert make_pipe(VLLM_SR_API_KEY="valve").config()["VLLM_SR_API_KEY"] == "valve"


def test_forward_body_effort():
    pipe = make_pipe(EFFORT_TIERS="opus,fable")
    plan = jr.Plan("opus", "large", "xhigh", "")
    assert pipe.forward_body({"model": "jev_router", "messages": []}, plan) == {"model": "large", "messages": [], "reasoning_effort": "high"}
    assert "reasoning_effort" not in pipe.forward_body({"model": "x"}, jr.Plan("sonnet", "medium", "high", ""))
    assert pipe.forward_body({"model": "x", "reasoning_effort": "low"}, plan)["reasoning_effort"] == "low"
    assert "reasoning_effort" not in pipe.forward_body({"model": "x"}, jr.Plan("opus", "large", None, ""))
    off = make_pipe(SEND_REASONING_EFFORT=False)
    assert "reasoning_effort" not in off.forward_body({"model": "x"}, plan)


def test_loop_guard():
    assert "OPUS_MODEL" in make_pipe(OPUS_MODEL="jev_router").loops_back("jev_router")
    assert make_pipe().loops_back("jev_router") is None


def install_fake_open_webui(monkeypatch, sent):
    async def generate_chat_completion(request, body, user):
        sent.append({"request": request, "body": body, "user": user})
        return {"choices": [{"message": {"content": "ok"}}]}

    class Users:
        @staticmethod
        async def get_user_by_id(uid):
            return {"id": uid}

    modules = {
        "open_webui": types.ModuleType("open_webui"),
        "open_webui.models": types.ModuleType("open_webui.models"),
        "open_webui.models.users": types.ModuleType("open_webui.models.users"),
        "open_webui.utils": types.ModuleType("open_webui.utils"),
        "open_webui.utils.chat": types.ModuleType("open_webui.utils.chat"),
    }
    modules["open_webui.models.users"].Users = Users
    modules["open_webui.utils.chat"].generate_chat_completion = generate_chat_completion
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_pipe_forwards_and_shows_status(monkeypatch):
    sent, events = [], []
    install_fake_open_webui(monkeypatch, sent)
    pipe = make_pipe()
    pipe._fetch = FakeFetch((200, jev_answer("opus", 0.9, 2)))

    async def emit(event):
        events.append(event)

    body = {"model": "jev_router", "messages": [{"role": "user", "content": "refactor this"}], "stream": True}
    out = run(pipe.pipe(body, __user__={"id": "u1"}, __request__="REQ", __event_emitter__=emit, __metadata__={"chat_id": "c1"}))
    assert out["choices"][0]["message"]["content"] == "ok"
    assert sent[0]["body"]["model"] == "large" and sent[0]["body"]["reasoning_effort"] == "high" and sent[0]["body"]["stream"] is True
    assert sent[0]["user"] == {"id": "u1"} and sent[0]["request"] == "REQ"
    assert events[0]["type"] == "status" and events[0]["data"]["description"].endswith("→ large")
    assert pipe._last_tier["c1"] == "opus"


def test_pipe_sends_tasks_to_the_lowest_tier_without_asking(monkeypatch):
    sent = []
    install_fake_open_webui(monkeypatch, sent)
    pipe = make_pipe()
    pipe._fetch = fetch = FakeFetch()
    run(pipe.pipe({"model": "jev_router", "messages": [{"role": "user", "content": "title?"}]}, __user__={"id": "u"}, __task__="title_generation"))
    assert fetch.calls == [] and sent[0]["body"]["model"] == "small" and "reasoning_effort" not in sent[0]["body"]


def test_pipe_without_models_says_what_to_fill_in():
    pipe = jr.Pipe()
    out = run(pipe.pipe({"model": "jev_router", "messages": []}))
    assert "HAIKU_MODEL" in out
