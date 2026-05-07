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

    /* Pixel Font */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    /* Main Background */
    .stApp {
        background-image: url("https://imgs.search.brave.com/4kwaWEiKKkhzl4O4ziHxZ3rSYpQ5FvBk9YD4xM6T6VM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRpYS5mb3JnZWNkbi5u/ZXQvYXR0YWNobWVu/dHMvNzYwLzEzLzIw/MjMtMTEtMTZfMTAu/cG5n");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Dark Overlay */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.45);
        z-index: -1;
    }

    /* Global Text Styling */
    html, body, [class*="css"] {
        font-family: 'Press Start 2P', cursive;
        color: white;
    }

    /* Title */
    h1 {
        color: #7CFC00 !important;
        text-shadow: 3px 3px 0px black;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2rem !important;
    }

    /* Chat Message Containers */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255,255,255,0.15);

        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);

        padding: 1rem;
        border-radius: 20px;

        margin-bottom: 1rem;

        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    /* User Bubble */
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: rgba(50, 205, 50, 0.18);
        border: 1px solid rgba(50,205,50,0.35);
    }

    /* Assistant Bubble */
    [data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.08);
    }

    /* Chat Input Area */
    .stChatInputContainer {
        background: rgba(0,0,0,0.55);
        border-top: 1px solid rgba(255,255,255,0.1);

        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* Input Box */
    .stChatInputContainer textarea {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;

        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;

        font-size: 12px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
    }

    /* Markdown text */
    p, div, span {
        color: white !important;
        line-height: 1.8;
        font-size: 11px;
    }

    /* Scrollbar */
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
    answer = data.get("answer", "No answer returned.")
    sources = data.get("retrieved_chunks", [])

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