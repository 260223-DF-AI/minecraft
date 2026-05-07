import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/research"

st.set_page_config(page_title="Minecraft RAG Assistant", layout="wide")

st.title("Minecraft RAG Assistant")

# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Render chat history
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User input
# -------------------------
user_input = st.chat_input("Let me pick your brain...")

if user_input:

    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------
    # Call backend
    # -------------------------
    with st.spinner("Steve is thinking..."):
        response = requests.post(
            API_URL,
            json={"question": user_input}
        )

        data = response.json()

    # -------------------------
    # Extract response safely
    # -------------------------
    answer = data.get("answer", "No answer returned.")
    sources = data.get("retrieved_chunks", [])

    # -------------------------
    # Show assistant message
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources"):
            for chunk in sources:
                metadata = chunk.get("metadata", {})
                st.markdown(f"**{metadata.get('source', 'unknown')}**")
                st.write(chunk.get("page_content", ""))
                st.divider()

    # -------------------------
    # Save assistant message
    # -------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })