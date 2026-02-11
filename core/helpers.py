import datetime as dt
import json
from typing import Any, Dict, Optional

import pandas as pd

from config import (
    LEGAL_HINTS,
    MENSAJE_NO_CLASIFICABLE,
    MENSAJE_SCORE_BAJO,
    NONLEGAL_HINTS,
    PROMPT_LEAK_PATTERNS,
    SUSPECT_PATTERNS,
)


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except Exception:
            return None
    return None


def extract_prefixed_value(text: str, prefix: str) -> Optional[str]:
    raw = (text or "").strip()
    if raw.upper().startswith(prefix.upper()):
        return raw[len(prefix) :].strip()
    return None


def is_no_relevante(raw_text: str) -> bool:
    return "NO_RELEVANTE" in (raw_text or "").strip().upper()


def looks_like_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in SUSPECT_PATTERNS)


def is_prompt_leak_attempt(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in PROMPT_LEAK_PATTERNS)


def fuera_de_ambito_rapido(text: str) -> bool:
    t = (text or "").lower()
    has_legal = any(w in t for w in LEGAL_HINTS)
    has_nonlegal = any(w in t for w in NONLEGAL_HINTS)
    return has_nonlegal and not has_legal


def format_no_clasificable_msg(hits: pd.DataFrame) -> str:
    if hits is None or hits.empty:
        return MENSAJE_SCORE_BAJO

    nombres = [
        str(row["Trámite"])
        for _, row in hits.head(3).iterrows()
        if pd.notna(row.get("Trámite"))
    ]
    if not nombres:
        return MENSAJE_SCORE_BAJO

    if len(nombres) == 1:
        lista = nombres[0]
    elif len(nombres) == 2:
        lista = f"{nombres[0]} y {nombres[1]}"
    else:
        lista = f"{', '.join(nombres[:-1])} y {nombres[-1]}"
    return MENSAJE_NO_CLASIFICABLE.format(categorias=lista)


def now_label() -> str:
    return dt.datetime.now().strftime("%H:%M")


def make_title_from_text(text: str, max_len: int = 34) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def usage_get(usage: Any, key: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def extract_usage_totals(usage: Any) -> Dict[str, int]:
    prompt_tokens = safe_int(usage_get(usage, "prompt_tokens") or usage_get(usage, "input_tokens"))
    completion_tokens = safe_int(
        usage_get(usage, "completion_tokens") or usage_get(usage, "output_tokens")
    )
    total_tokens = safe_int(usage_get(usage, "total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def new_query_token_trace() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": [],
    }


def track_usage(trace: Optional[Dict[str, Any]], label: str, usage: Any) -> None:
    if trace is None or usage is None:
        return

    totals = extract_usage_totals(usage)
    if totals["prompt_tokens"] == 0 and totals["completion_tokens"] == 0 and totals["total_tokens"] == 0:
        return

    trace["prompt_tokens"] += totals["prompt_tokens"]
    trace["completion_tokens"] += totals["completion_tokens"]
    trace["total_tokens"] += totals["total_tokens"]
    trace["calls"].append(
        {
            "label": label,
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
        }
    )


def attach_token_trace_to_last_assistant_message(conversation: Dict[str, Any], trace: Dict[str, Any]) -> None:
    if not trace:
        return
    if trace.get("total_tokens", 0) == 0 and not trace.get("calls"):
        return

    for msg in reversed(conversation.get("messages", [])):
        if msg.get("role") == "assistant":
            msg["token_usage"] = {
                "prompt_tokens": trace.get("prompt_tokens", 0),
                "completion_tokens": trace.get("completion_tokens", 0),
                "total_tokens": trace.get("total_tokens", 0),
                "calls": list(trace.get("calls", [])),
            }
            return
