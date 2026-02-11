import datetime as dt
import json


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
