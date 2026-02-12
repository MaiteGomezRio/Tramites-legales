import datetime as dt
import json
import streamlit as st


def try_parse_json(text: str):
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


def badge_for_conf(conf):
    if conf is None:
        return '<span class="badge badge-no">sin confianza</span>'
    if conf >= 0.75:
        return f'<span class="badge badge-ok">confianza alta · {conf:.2f}</span>'
    if conf >= 0.55:
        return f'<span class="badge badge-warn">confianza media · {conf:.2f}</span>'
    return f'<span class="badge badge-no">confianza baja · {conf:.2f}</span>'


def now_label():
    return dt.datetime.now().strftime("%H:%M")


def make_title_from_text(text: str, max_len=34):
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def normalize_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def build_usage_summary(agent_usages):
    if not agent_usages:
        return None

    agents = []
    total_input = 0
    total_output = 0
    total_tokens = 0
    has_any_total = False

    for agent_name, usage in agent_usages:
        usage = usage or {}
        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        tot_t = usage.get("total_tokens")

        agents.append(
            {
                "agent": agent_name,
                "input_tokens": in_t,
                "output_tokens": out_t,
                "total_tokens": tot_t,
            }
        )

        if isinstance(in_t, int):
            total_input += in_t
            has_any_total = True
        if isinstance(out_t, int):
            total_output += out_t
            has_any_total = True
        if isinstance(tot_t, int):
            total_tokens += tot_t
            has_any_total = True

    summary = {"agents": agents}
    if has_any_total:
        summary["total"] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
        }
    return summary


def render_token_usage_expander(usage):
    with st.expander("Tokens usados"):
        if not usage:
            st.caption("No disponible")
            return

        if isinstance(usage, dict) and "agents" in usage:
            st.caption("Desglose por agente: ")
            for agent_usage in usage["agents"]:
                st.caption(f"- {agent_usage.get('agent', 'agente')}: Entrada {agent_usage.get('input_tokens', 'N/D')} | Salida {agent_usage.get('output_tokens', 'N/D')} | Total {agent_usage.get('total_tokens', 'N/D')}")

            total = usage.get("total")
            if total:
                st.caption(f"Total de la consulta: Entrada {total.get('input_tokens', 'N/D')} | Salida {total.get('output_tokens', 'N/D')} | Total {total.get('total_tokens', 'N/D')}")
            return

        st.caption(
            f"Entrada: {usage.get('input_tokens', 'N/D')} | "
            f"Salida: {usage.get('output_tokens', 'N/D')} | "
            f"Total: {usage.get('total_tokens', 'N/D')}"
        )
