import os
os.environ['TORCH_CLASSES_IGNORE_MISSING'] = '1'

import torch
import streamlit as st
import pandas as pd
from datetime import datetime

# === HARD-CODE YOUR API KEY HERE ===
GEMINI_API_KEY = "AIzaSyA-Y1bbM0aofvo_roKegn3Z_37eAw2ZpWc"
TAVILY_API_KEY = None  # Optional, can be set if needed

# Import math agent
try:
    from math_agent import MathAgent, create_math_agent
except ImportError:
    st.error("Could not import math_agent. Make sure math_agent.py is in the same directory.")
    st.stop()

# Streamlit config
st.set_page_config(
    page_title="Math Tutor AI",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# State
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'math_agent' not in st.session_state:
        st.session_state.math_agent = None
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []

@st.cache_resource
def get_math_agent():
    try:
        return create_math_agent(GEMINI_API_KEY, TAVILY_API_KEY)
    except Exception as e:
        st.error(f"Error creating math agent: {str(e)}")
        return None

def display_chat_message(role, content, timestamp=None):
    with st.chat_message(role):
        st.markdown(content)
        if timestamp:
            st.caption(f"⏰ {timestamp}")

def save_feedback(question, answer, feedback):
    data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': answer,
        'feedback': feedback
    }
    st.session_state.feedback_log.append(data)
    try:
        df = pd.DataFrame([data])
        if os.path.exists("feedback_log.csv"):
            df.to_csv("feedback_log.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("feedback_log.csv", mode='w', header=True, index=False)
    except:
        pass

def main():
    init_session_state()
    st.markdown("<h1 class='main-header'>🧮 Math Tutor AI</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # System Controls
    with st.sidebar:
        st.subheader("🔧 Controls")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("♻️ Reset Agent"):
            st.session_state.math_agent = None
            st.cache_resource.clear()
            st.success("Agent reset successfully!")

        st.subheader("📊 Stats")
        st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("Feedbacks", len(st.session_state.feedback_log))

        if st.session_state.feedback_log:
            if st.button("📥 Download Feedback"):
                df = pd.DataFrame(st.session_state.feedback_log)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Initialize agent
    if st.session_state.math_agent is None:
        with st.spinner("Initializing AI agent..."):
            st.session_state.math_agent = get_math_agent()

    if not st.session_state.math_agent:
        st.stop()

    # Show chat history
    for i, msg in enumerate(st.session_state.messages):
        display_chat_message(msg["role"], msg["content"])
        if msg["role"] == "assistant":
            col1, col2, _ = st.columns([1, 1, 4])
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    user_msg = st.session_state.messages[i-1]["content"]
                    save_feedback(user_msg, msg["content"], "helpful")
                    st.success("Thanks! 👍")
            with col2:
                if st.button("👎", key=f"dislike_{i}"):
                    user_msg = st.session_state.messages[i-1]["content"]
                    save_feedback(user_msg, msg["content"], "not_helpful")
                    st.info("We'll improve 👎")

    # User input
    user_input = st.text_input("💬 Ask a math question:")
    if st.button("Ask") and user_input:
        display_chat_message("user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Solving..."):
                try:
                    result = st.session_state.math_agent.get_answer(user_input)
                    st.write("Debug result:", result)  # Remove this after testing
                    response = result.get("answer", "❌ No answer.")
                    st.markdown(response)

                    if result.get("context_used"):
                        with st.expander("📚 Context Used"):
                            st.text(result.get("context", "No context"))

                    if result.get("success"):
                        st.success("✅ Answer generated!")
                    else:
                        st.warning("⚠️ Double-check the answer.")

                except Exception as e:
                    response = f"❌ Error: {str(e)}"
                    st.error(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.8em; color: gray'>
        Math Tutor AI | Built with Gemini + GSM8K | For academic help only
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
