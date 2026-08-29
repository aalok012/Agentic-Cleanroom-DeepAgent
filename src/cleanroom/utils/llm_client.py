import json
import os
import re

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.cleanroom.llms.callbacks.metric import GLOBAL_HANDLER

load_dotenv(override=True)

# The pipeline talks to any OpenAI-compatible /v1 endpoint — a self-hosted vLLM / SGLang /
# llama.cpp / TGI / Ollama server, or api.openai.com. Point it at yours with LLM_BASE_URL
# (OPENAI_BASE_URL is still honoured as an alias):
#
#   LLM_BASE_URL=http://localhost:8000/v1                 # tunnel to the Minsky vLLM job
#   LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
#   LLM_API_KEY=EMPTY          # most self-hosted servers ignore the key
#
# scripts/minsky/tunnel.sh writes that .env for you from the running Slurm job.
#
# Model ids are passed through VERBATIM — use exactly what your server's /v1/models lists.
DEFAULT_LOCAL_API_KEY = "EMPTY"

# Default models for the normal pipeline and the Dafny/proof stage. Override per-run with
# --model, or globally with LLM_MODEL / LLM_DAFNY_MODEL in .env.
# Qwen2.5-Coder-32B in its official 4-bit AWQ build — the largest coder model that fits
# Minsky's single 48 GB GPU. See scripts/minsky/models.conf.
DEFAULT_MODEL = os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
DAFNY_MODEL = os.getenv("LLM_DAFNY_MODEL") or DEFAULT_MODEL

_DEBUG_ACTIVE_RUN_ID = None
_DEBUG_INPUT_PRINTED = False
_DEBUG_OUTPUT_PRINTED = False


def _debug_once_enabled() -> bool:
    return os.getenv("CLEANROOM_LLM_DEBUG_ONCE", "").lower() in ("1", "true", "yes", "on")


def _llm_timeout() -> float:
    """Per-request HTTP timeout (seconds) for the LLM client. A bounded timeout turns a hung /
    half-delivered response (the transient upstream `JSONDecodeError` we hit) into a clean
    timeout the SDK can RETRY instead of crashing the whole run.

    Default 120s — generous enough for the Dafny/proof stage's long reasoning responses while
    still catching hangs. Lower it (e.g. 30) via CLEANROOM_LLM_TIMEOUT for snappier failure on
    the lighter stages."""
    try:
        return max(5.0, float(os.getenv("CLEANROOM_LLM_TIMEOUT", "120")))
    except ValueError:
        return 120.0


def _llm_max_retries() -> int:
    """Automatic retries the OpenAI SDK performs on transient failures (timeouts, connection
    errors, 429 rate limits, 5xx). Default 3 (SDK default is 2); override with
    CLEANROOM_LLM_MAX_RETRIES."""
    try:
        return max(0, int(os.getenv("CLEANROOM_LLM_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def _debug_max_chars() -> int:
    try:
        return max(500, int(os.getenv("CLEANROOM_LLM_DEBUG_MAX_CHARS", "4000")))
    except ValueError:
        return 4000


def _truncate(text: str, limit: int | None = None) -> str:
    limit = _debug_max_chars() if limit is None else limit
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _safe_json(value) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return repr(value)


def _message_debug_dict(msg) -> dict:
    data = {
        "type": getattr(msg, "type", type(msg).__name__),
        "content": getattr(msg, "content", None),
    }
    for attr in ("name", "tool_calls", "invalid_tool_calls", "additional_kwargs",
                 "response_metadata", "usage_metadata"):
        value = getattr(msg, attr, None)
        if value:
            data[attr] = value
    return data


def _find_reasoning_fields(value, prefix: str = "") -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "reason" in str(key).lower():
                fields.append((path, child))
            fields.extend(_find_reasoning_fields(child, path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            fields.extend(_find_reasoning_fields(child, f"{prefix}[{i}]"))
    return fields


class _OneShotLLMDebugCallback(BaseCallbackHandler):
    """Print one prompt and its raw model response before structured-output parsing."""

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        global _DEBUG_ACTIVE_RUN_ID, _DEBUG_INPUT_PRINTED
        if _DEBUG_INPUT_PRINTED or _DEBUG_OUTPUT_PRINTED:
            return
        _DEBUG_ACTIVE_RUN_ID = run_id
        _DEBUG_INPUT_PRINTED = True
        serial = serialized.get("name") if isinstance(serialized, dict) else None
        print("\n=== CLEANROOM LLM DEBUG: first input before structured output ===")
        if serial:
            print(f"model wrapper: {serial}")
        payload = [[_message_debug_dict(msg) for msg in batch] for batch in messages]
        print(_truncate(_safe_json(payload)))

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        global _DEBUG_ACTIVE_RUN_ID, _DEBUG_INPUT_PRINTED
        if _DEBUG_INPUT_PRINTED or _DEBUG_OUTPUT_PRINTED:
            return
        _DEBUG_ACTIVE_RUN_ID = run_id
        _DEBUG_INPUT_PRINTED = True
        serial = serialized.get("name") if isinstance(serialized, dict) else None
        print("\n=== CLEANROOM LLM DEBUG: first input before structured output ===")
        if serial:
            print(f"model wrapper: {serial}")
        print(_truncate(_safe_json(prompts)))

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        global _DEBUG_ACTIVE_RUN_ID, _DEBUG_OUTPUT_PRINTED
        if _DEBUG_OUTPUT_PRINTED:
            return
        if _DEBUG_ACTIVE_RUN_ID is not None and run_id != _DEBUG_ACTIVE_RUN_ID:
            return
        _DEBUG_OUTPUT_PRINTED = True
        _DEBUG_ACTIVE_RUN_ID = run_id

        generations = []
        usage_payloads = []
        for batch in getattr(response, "generations", None) or []:
            batch_items = []
            for gen in batch:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    item = _message_debug_dict(msg)
                    usage_payloads.append(item.get("usage_metadata") or {})
                    usage_payloads.append(item.get("response_metadata") or {})
                else:
                    item = {"text": getattr(gen, "text", None)}
                info = getattr(gen, "generation_info", None)
                if info:
                    item["generation_info"] = info
                    usage_payloads.append(info)
                batch_items.append(item)
            generations.append(batch_items)

        llm_output = getattr(response, "llm_output", None) or {}
        usage_payloads.append(llm_output)
        debug_payload = {
            "generations": generations,
            "llm_output": llm_output,
        }
        reasoning_fields = []
        for payload in usage_payloads:
            reasoning_fields.extend(_find_reasoning_fields(payload))
        deduped_reasoning_fields = []
        seen_reasoning_fields = set()
        for path, value in reasoning_fields:
            key = (path, repr(value))
            if key not in seen_reasoning_fields:
                seen_reasoning_fields.add(key)
                deduped_reasoning_fields.append((path, value))

        print("\n=== CLEANROOM LLM DEBUG: first raw output before structured parsing ===")
        print(_truncate(_safe_json(debug_payload)))
        print("\n=== CLEANROOM LLM DEBUG: reasoning-token fields ===")
        if deduped_reasoning_fields:
            for path, value in deduped_reasoning_fields:
                print(f"{path}: {value}")
        else:
            print("no usage fields containing 'reason' were returned")


def _extract_json(text: str | None) -> dict | None:
    """Best-effort pull of a JSON object out of free text — strips ```json fences and surrounding
    prose, then loads the outermost {...}. Returns None if nothing parseable is found."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        t = t[start : end + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_structured(d: dict, schema):
    """Turn the `include_raw=True` envelope ({raw, parsed, parsing_error}) back into the parsed
    object — but if the tool call was absent or failed to validate, recover the data from the
    message's tool-call args or its text content (some models, esp. 'thinking' ones, answer in
    prose/fenced JSON instead of emitting a clean tool call). Returns None only if nothing is
    recoverable, so callers that tolerate emptiness (e.g. dependency edges) still see None."""
    parsed = d.get("parsed") if isinstance(d, dict) else None
    if parsed is not None:
        return parsed                                  # normal path — no behavior change
    if not hasattr(schema, "model_validate"):
        return parsed
    raw = d.get("raw") if isinstance(d, dict) else None
    # 1) a tool call was made but its args failed strict validation — try a lenient re-validate
    for tc in (getattr(raw, "tool_calls", None) or []):
        args = tc.get("args") if isinstance(tc, dict) else None
        if isinstance(args, dict):
            try:
                return schema.model_validate(args)
            except Exception:
                pass
    # 2) no usable tool call — recover JSON from the message content
    content = getattr(raw, "content", None)
    text = content if isinstance(content, str) else (
        "".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else None
    )
    obj = _extract_json(text)
    if obj is not None:
        try:
            return schema.model_validate(obj)
        except Exception:
            pass
    return None


class _ChatOpenAI(ChatOpenAI):
    """ChatOpenAI hardened for the mixed-model pipeline. Two changes to `with_structured_output`:

    1. Default method is `function_calling`, not langchain's `json_schema`. OpenAI/Anthropic honor
       strict json_schema, but open-weight models often wrap the JSON in a ```json fence,
       which the strict parser rejects. Tool-calling args are provider-normalized and work for all.
    2. We request `include_raw=True` under the hood and post-process: if the model emits no clean
       tool call (common for 'thinking' models that answer in prose), we recover the structured
       data from its tool-call args or message content instead of returning None. The agent still
       receives the parsed pydantic object — the recovery is invisible to callers.

    A caller passing `include_raw=True` itself gets the raw envelope untouched; an explicit
    `method=` overrides the default.
    """

    def with_structured_output(self, schema=None, *, method="function_calling",
                               include_raw=False, **kwargs):
        base = super().with_structured_output(
            schema, method=method, include_raw=True, **kwargs)
        if include_raw:
            return base                                # caller asked for the raw envelope

        def _coerce_with_retry(inp, _s=schema, _base=base):
            # _coerce_structured returns None when a model reply is unparseable (no clean tool
            # call AND no recoverable JSON/text) — which happens intermittently (provider
            # non-determinism). Re-invoke the structured call a few times before giving up so a
            # transient None becomes a clean object instead of crashing whatever agent
            # dereferences it (.content / .cases / …). Returns None only if every attempt fails.
            out = _coerce_structured(_base.invoke(inp), _s)
            attempts = 0
            while out is None and attempts < 3:
                attempts += 1
                out = _coerce_structured(_base.invoke(inp), _s)
            return out
        return RunnableLambda(_coerce_with_retry)


def llm_base_url() -> str | None:
    """Base URL of the OpenAI-compatible endpoint, or None for api.openai.com."""
    return os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None


def llm_api_key() -> str | None:
    """API key for ChatOpenAI. Self-hosted servers usually ignore it, so when a custom
    base URL is configured we fall back to a placeholder instead of failing."""
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key and llm_base_url():
        return DEFAULT_LOCAL_API_KEY
    return key


def llm_api_key_configured() -> bool:
    return bool(llm_api_key())


def resolve_model(name: str) -> str:
    """Model ids are passed through verbatim — whatever the endpoint's /v1/models advertises.
    LLM_MODEL_PREFIX still prepends a namespace if your server expects one."""
    prefix = os.getenv("LLM_MODEL_PREFIX", "")
    if prefix and not name.startswith(prefix):
        return f"{prefix}{name}"
    return name


def _floor_temperature(temperature: float) -> float:
    """Reasoning models (e.g. DeepSeek-R1 distills) degenerate into repetition at
    temperature 0; DeepSeek recommend ~0.6. CLEANROOM_LLM_TEMPERATURE raises the
    deterministic default for such a model without touching the certification stage,
    which passes its own higher temperature to draw diverse pass@k samples."""
    if temperature != 0.0:
        return temperature
    try:
        return float(os.getenv("CLEANROOM_LLM_TEMPERATURE", "") or 0.0)
    except ValueError:
        return 0.0


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatOpenAI:
    # temperature defaults to 0 (deterministic) for the generation/judging stages;
    # the certification stage raises it to draw diverse samples for pass@k.
    temperature = _floor_temperature(temperature)
    # GLOBAL_HANDLER counts tokens/latency for every call (counts only, no content).
    base_url = llm_base_url()
    api_key = llm_api_key()

    callbacks = [GLOBAL_HANDLER]
    if _debug_once_enabled():
        callbacks.append(_OneShotLLMDebugCallback())

    resolved_model = resolve_model(model)

    kwargs: dict = {
        "model": resolved_model,
        "temperature": temperature,
        "api_key": api_key,
        "callbacks": callbacks,
        # Bounded per-request timeout + automatic retries so a transient endpoint hiccup
        # (timeout / 429 / 5xx / half-delivered body → JSONDecodeError) is retried instead of
        # killing the whole run. Both env-tunable (CLEANROOM_LLM_TIMEOUT / _MAX_RETRIES).
        "timeout": _llm_timeout(),
        "max_retries": _llm_max_retries(),
    }
    if base_url:
        kwargs["base_url"] = base_url

    # _ChatOpenAI defaults structured output to function/tool calling (see class docstring) so
    # open-weight models served locally work; for OpenAI it behaves the same as plain ChatOpenAI.
    return _ChatOpenAI(**kwargs)
