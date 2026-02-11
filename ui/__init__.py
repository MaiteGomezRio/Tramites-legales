import streamlit as st

from config import LOGO_SIDEBAR
from conversations import create_and_switch, delete_conversation, get_active, new_conversation, set_active


def load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.image(LOGO_SIDEBAR, width="stretch")
        st.markdown("## Conversaciones")

        if st.button("➕ Nuevo chat", use_container_width=True):
            create_and_switch()
            st.rerun()

        st.write("")

        for c in sorted(st.session_state.conversations, key=lambda x: x["created_at"], reverse=True):
            is_active = c["id"] == st.session_state.active_id
            title = c["title"] or "Chat"
            suffix = " ✅" if c.get("closed") else ""
            label = f"{'👉 ' if is_active else ''}{title}{suffix}"
            if st.button(label, key=f"open_{c['id']}", use_container_width=True):
                set_active(c["id"])
                st.rerun()

        st.write("")
        st.markdown("---")
        active = get_active()
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Borrar chat", use_container_width=True):
                delete_conversation(active["id"])
                st.rerun()
        with col_b:
            if st.button("Borrar todo", use_container_width=True):
                st.session_state.conversations = [new_conversation(seed_assistant=True)]
                st.session_state.active_id = st.session_state.conversations[0]["id"]
                st.rerun()


def render_debug_info(hits, umbral_ambiguedad):
    if hits is None or len(hits) == 0:
        st.markdown("Sin candidatos para mostrar.")
        return

    top_score = hits.iloc[0]["score"]
    st.markdown("**Candidatos evaluados:**")

    for _, row in hits.iterrows():
        gap = top_score - row["score"]
        marker = " ⚠️ ambiguo" if gap < umbral_ambiguedad else ""
        st.markdown(f"- `{row['score']:.4f}` · **{row['Trámite']}**{marker}")


def render_chat_messages(active, assistant_avatar: str, user_avatar: str):
    for msg in active["messages"]:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        avatar = assistant_avatar if role == "assistant" else user_avatar

        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])
            if msg.get("t"):
                st.caption(msg["t"])

            if role == "assistant" and msg.get("token_usage"):
                tu = msg["token_usage"]
                st.caption(
                    f"Tokens consulta: prompt={tu.get('prompt_tokens', 0)} | "
                    f"completion={tu.get('completion_tokens', 0)} | "
                    f"total={tu.get('total_tokens', 0)}"
                )
                calls = tu.get("calls") or []
                if calls:
                    with st.expander("Ver tokens por llamada"):
                        for call in calls:
                            st.write(
                                f"{call.get('label', 'llamada')}: "
                                f"prompt={call.get('prompt_tokens', 0)} | "
                                f"completion={call.get('completion_tokens', 0)} | "
                                f"total={call.get('total_tokens', 0)}"
                            )

            if role == "assistant" and "debug" in msg:
                with st.expander("ℹ️ Ver detalles de clasificación"):
                    render_debug_info(
                        hits=msg["debug"]["hits"],
                        umbral_ambiguedad=msg["debug"]["umbral"],
                    )
