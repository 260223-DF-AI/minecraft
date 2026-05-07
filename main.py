import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/research"

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Minecraft RAG Assistant",
    layout="wide"
)

# -------------------------------------------------
# Custom Minecraft Styling
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* =====================================================
       PIXEL FONT
    ===================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    html, body, [class*="css"] {
        font-family: 'Press Start 2P', cursive !important;
        color: white !important;
    }

    /* FORCE FONT INSIDE EVERYTHING (IMPORTANT FIX) */
    * {
        font-family: 'Press Start 2P', cursive !important;
    }

    /* =====================================================
       BACKGROUND
    ===================================================== */
    [data-testid="stAppViewContainer"] {
        background: url("https://imgs.search.brave.com/4kwaWEiKKkhzl4O4ziHxZ3rSYpQ5FvBk9YD4xM6T6VM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRpYS5mb3JnZWNkbi5u/ZXQvYXR0YWNobWVu/dHMvNzYwLzEzLzIw/MjMtMTEtMTZfMTAu/cG5n");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* dark overlay */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }

    .main {
        position: relative;
        z-index: 1;
    }

    /* =====================================================
       TITLE
    ===================================================== */
    h1 {
        color: #7CFC00 !important;
        text-shadow: 3px 3px 0 black;
        text-align: center;
        font-size: 24px !important;
    }

    /* =====================================================
       CHAT BUBBLES (GLASS)
    ===================================================== */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;

        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);

        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;

        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    /* user bubble */
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: rgba(50,205,50,0.15) !important;
        border: 1px solid rgba(50,205,50,0.35) !important;
    }

    /* assistant bubble */
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.07) !important;
    }

    /* =====================================================
       FORCE TEXT INSIDE CHAT (CRITICAL FIX)
    ===================================================== */
    [data-testid="stChatMessage"] * {
        font-family: 'Press Start 2P', cursive !important;
        color: white !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        font-family: 'Press Start 2P', cursive !important;
        color: white !important;
    }

    /* =====================================================
       CHAT INPUT
    ===================================================== */
    [data-testid="stChatInput"] {
        background: rgba(0,0,0,0.6) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    [data-testid="stChatInput"] textarea {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 10px !important;

        background: rgba(255,255,255,0.08) !important;
        color: white !important;

        border: 1px solid rgba(124,252,0,0.4) !important;
        border-radius: 10px !important;

        padding: 10px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        font-family: 'Press Start 2P', cursive !important;
        color: rgba(255,255,255,0.6) !important;
    }

    /* =====================================================
       EXPANDER
    ===================================================== */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
    }

    /* =====================================================
       SCROLLBAR
    ===================================================== */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
    }

    ::-webkit-scrollbar-thumb {
        background: #7CFC00;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("Minecraft RAG Assistant")

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Render Chat History
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# User Input
# -------------------------------------------------
user_input = st.chat_input("Let me pick your brain...")

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------------------------
    # Backend Request
    # -------------------------------------------------
    with st.spinner("Steve is mining for answers..."):
        response = requests.post(
            API_URL,
            json={"question": user_input}
        )

        data = response.json()
        print(data)
    # -------------------------------------------------
    # Extract Response
    # -------------------------------------------------
    answer = data["analysis"]["answer"]
    sources = data["analysis"]["citations"]

    # -------------------------------------------------
    # Assistant Message
    # -------------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources"):
            for chunk in sources:
                metadata = chunk.get("metadata", {})

                st.markdown(
                    f"### {metadata.get('source', 'unknown')}"
                )

                st.write(
                    chunk.get("page_content", "")
                )

                st.divider()

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })