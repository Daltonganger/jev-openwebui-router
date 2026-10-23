"""
Asks the decision models the router would ask, outside Open WebUI, and prints
what each one says about a prompt. Settings come from the environment, under
the valves' names (PROVIDERS, VLLM_SR_API_KEY, OPENROUTER_API_KEY, ...).

    PROVIDERS=vllm,openrouter VLLM_SR_API_KEY=... OPENROUTER_API_KEY=... \
      python scripts/check_decider.py "plan a database migration"

Each route is asked on its own (not as a chain), so a list shows all of them.
Needs aiohttp (pip install aiohttp pydantic).
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jev_router as jr  # noqa: E402


async def main(prompt: str) -> int:
    cfg = {name: os.environ.get(name, "") for name in jr.Pipe.Valves.model_fields}
    cfg["ALLOW_CUSTOM_BASE"] = jr.flag_on(cfg.get("ALLOW_CUSTOM_BASE"))
    routes, why = jr.routes_of(cfg)
    if why:
        print(f"not set up: {why}")
        return 2
    offered = [t for t in jr.TIERS if t not in jr.tier_list_of(os.environ.get("EXCLUDE", ""))]
    timeout_ms = jr.timeout_of(os.environ.get("TIMEOUT_MS") or 5000)
    failed = 0
    for route in routes:
        result = await jr.ask_chain([route], prompt, offered, timeout_ms, jr.aiohttp_fetch)
        if not result["ok"]:
            print(f"{route.name:21} {route.model:32} failed: {result['reason']}")
            failed += 1
            continue
        d = jr.decision_of(result["answers"], offered, jr.confidence_source_of(route), jr.label_of(route))
        if d is None:
            print(f"{route.name:21} {route.model:32} unusable answer: {result['answers']}")
            failed += 1
            continue
        probs = " ".join(f"{t}={p:.2f}" for t, p in (d.probabilities or {}).items())
        print(f"{route.name:21} {route.model:32} {d.tier:6} {d.effort:6} {d.decider} {jr.pct(d.confidence)} {result['ms']}ms  {probs}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(" ".join(sys.argv[1:]) or "rename this variable to snake_case")))
