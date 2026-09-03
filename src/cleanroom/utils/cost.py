"""Rough USD cost estimate from token counts.

Prices are USD per 1M tokens and may drift over time — update PRICING if they change.
For models served on your own hardware there is no per-token bill; the estimate is only
meaningful for hosted APIs, so treat it as an upper bound on a self-hosted run.
"""

from src.cleanroom.utils.llm_client import DEFAULT_MODEL

PRICING: dict[str, dict[str, float]] = {
    # Self-hosted on Minsky — no per-token bill. Kept at 0 so the cost column stays
    # present (and honest) for runs served from our own GPU.
    "qwen2.5-coder-32b-instruct-awq": {"input": 0.0, "output": 0.0},
    "qwen2.5-coder-32b-instruct": {"input": 0.0, "output": 0.0},
    "qwen3-32b-awq": {"input": 0.0, "output": 0.0},
    "qwen3-32b-fp8": {"input": 0.0, "output": 0.0},
    "qwen3-32b": {"input": 0.0, "output": 0.0},
    "deepseek-r1-distill-qwen-32b": {"input": 0.0, "output": 0.0},
    # Hosted APIs used for the archived baselines in results/raw_results/.
    "deepseek-v3.2": {"input": 0.2288, "output": 0.3432},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}


def _normalize_model(model: str) -> str:
    """Strip a namespace prefix and casefold (``deepseek-ai/DeepSeek-V3.2`` → ``deepseek-v3.2``),
    so HuggingFace-style ids from a self-hosted server still match PRICING."""
    return (model.split("/", 1)[1] if "/" in model else model).lower()


# An unknown model bills nothing. Every self-hosted endpoint is free at the token level, and a
# missing price must never be able to raise: estimate_cost() is called from run_pipeline.py's
# FAILURE handler, so a KeyError here destroys the partial run record of an already-failing run
# and the run disappears from the metrics CSV entirely.
_FREE: dict[str, float] = {"input": 0.0, "output": 0.0}


def _rates(model: str) -> dict[str, float]:
    """Prices for a model, matching the longest known prefix (handles dated suffixes).

    Falls back to DEFAULT_MODEL's rates, then to free — never raises for an unpriced model."""
    model = _normalize_model(model)
    if model in PRICING:
        return PRICING[model]
    for name in sorted(PRICING, key=len, reverse=True):
        if model.startswith(name):
            return PRICING[name]
    return PRICING.get(_normalize_model(DEFAULT_MODEL), _FREE)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _rates(model)
    return input_tokens / 1_000_000 * rates["input"] + output_tokens / 1_000_000 * rates["output"]


def estimate_cost_by_model(calls) -> tuple[float, dict[str, dict]]:
    """Accurate cost for a MIXED-model run (e.g. --prove: gpt-4o-mini + gpt-4.1).

    Groups per-call records (``{model, input_tokens, output_tokens}`` from GLOBAL_METRICS.calls)
    by model and prices each group at its own rate. Returns (total_usd, {model: {input, output,
    calls, cost_usd}}). Pricing each model separately avoids the under/over-billing you get from
    applying one model's rate to every token.
    """
    by_model: dict[str, dict] = {}
    for c in calls or []:
        m = c.get("model") or DEFAULT_MODEL
        agg = by_model.setdefault(m, {"input": 0, "output": 0, "calls": 0, "cost_usd": 0.0})
        agg["input"] += int(c.get("input_tokens", 0))
        agg["output"] += int(c.get("output_tokens", 0))
        agg["calls"] += 1
    total = 0.0
    for m, agg in by_model.items():
        agg["cost_usd"] = round(estimate_cost(m, agg["input"], agg["output"]), 6)
        total += agg["cost_usd"]
    return round(total, 6), by_model
