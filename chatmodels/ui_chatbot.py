import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Mood Bot",
    page_icon="🎭",
    layout="centered",
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.title-container {
    text-align: center;
    padding: 10px 0 20px 0;
}
.title-container h1 {
    font-size: 2.2rem;
    margin-bottom: 0;
}
.title-container p {
    color: #9aa0a6;
    margin-top: 4px;
}
.mode-card {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="title-container">
    <h1>🎭 Mood Bot</h1>
    <p>Powered by LangChain + ChatMistralAI</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Model ----------------
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2603")

model = get_model()

# ---------------- Mode Definitions (same logic as original if/elif) ----------------
MODES = {
    "1": {
        "label": "😡 Angry Bot",
        "prompt": "You are an angry bot. Respond to the user in a very angry and aggressive manner.",
    },
    "2": {
        "label": "😂 Funny Bot",
        "prompt": "You are a funny bot. Respond to the user with humor and wit.",
    },
    "3": {
        "label": "😢 Sad Bot",
        "prompt": "You are a sad bot. Respond to the user in a very sad and melancholic manner.",
    },
}

# ---------------- Session State ----------------
if "mode_choice" not in st.session_state:
    st.session_state.mode_choice = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Mode Selection Screen ----------------
if st.session_state.mode_choice is None:
    st.subheader("Choose a mode:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("😡 Angry Bot", use_container_width=True):
            st.session_state.mode_choice = "1"
    with col2:
        if st.button("😂 Funny Bot", use_container_width=True):
            st.session_state.mode_choice = "2"
    with col3:
        if st.button("😢 Sad Bot", use_container_width=True):
            st.session_state.mode_choice = "3"

    if st.session_state.mode_choice is not None:
        mode_prompt = MODES[st.session_state.mode_choice]["prompt"]
        st.session_state.messages = [SystemMessage(content=mode_prompt)]
        st.rerun()

# ---------------- Chat Screen ----------------
else:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown(f"**Current Mode:** {MODES[st.session_state.mode_choice]['label']}")
        st.markdown("---")
        if st.button("🔄 Change Mode", use_container_width=True):
            st.session_state.mode_choice = None
            st.session_state.messages = []
            st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            mode_prompt = MODES[st.session_state.mode_choice]["prompt"]
            st.session_state.messages = [SystemMessage(content=mode_prompt)]
            st.rerun()

    st.markdown(f"### {MODES[st.session_state.mode_choice]['label']}")

    # Render chat history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)
        # SystemMessage intentionally not displayed

    prompt = st.chat_input("Ask me anything...")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = model.invoke(st.session_state.messages)
            st.markdown(response.content)

        st.session_state.messages.append(AIMessage(content=response.content))