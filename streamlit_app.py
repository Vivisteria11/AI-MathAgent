import os
os.environ['TORCH_CLASSES_IGNORE_MISSING'] = '1'

# Alternative: Set Streamlit to ignore torch.classes in file watching
import torch
# Disable file watcher
import streamlit as st
import pandas as pd
from datetime import datetime
import os



# Import your math agent (assuming it's in math_agent.py)
try:
    from math_agent import MathAgent, create_math_agent
except ImportError:
    st.error("Could not import math_agent. Make sure math_agent.py is in the same directory.")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Math Tutor AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'math_agent' not in st.session_state:
        st.session_state.math_agent = None
    
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []

@st.cache_resource
def get_math_agent(gemini_key, search_key=None):
    """Create and cache math agent instance"""
    try:
        agent = create_math_agent(gemini_key, search_key)
        return agent
    except Exception as e:
        st.error(f"Error creating math agent: {str(e)}")
        return None

def display_chat_message(role, content, timestamp=None):
    """Display a chat message with proper styling"""
    with st.chat_message(role):
        st.markdown(content)
        if timestamp:
            st.caption(f"⏰ {timestamp}")

def save_feedback_to_session(question, answer, feedback):
    """Save feedback to session state and try to save to file"""
    feedback_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': answer,
        'feedback': feedback
    }
    
    # Add to session state
    st.session_state.feedback_log.append(feedback_data)
    
    # Try to save to file (may not work in some cloud environments)
    try:
        df = pd.DataFrame([feedback_data])
        if os.path.exists("feedback_log.csv"):
            df.to_csv("feedback_log.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("feedback_log.csv", mode='w', header=True, index=False)
    except Exception:
        pass  # Fail silently if file operations don't work

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>🧮 Math Tutor AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        gemini_key = st.text_input(
            "🔑 Gemini API Key", 
            type="password", 
            help="Required for generating math solutions",
            placeholder="Enter your Gemini API key"
        )
        
        search_key = st.text_input(
            "🔍 Tavily API Key (Optional)", 
            type="password", 
            help="Optional for web search fallback",
            placeholder="Enter your Tavily API key (optional)"
        )
        
        st.markdown("---")
        
        # System controls
        st.subheader("🔧 System Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("♻️ Reset Agent"):
            st.session_state.math_agent = None
            st.cache_resource.clear()
            st.success("Agent reset successfully!")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Messages Sent", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("Feedback Received", len(st.session_state.feedback_log))
        
        # Export feedback
        if st.session_state.feedback_log:
            if st.button("📥 Download Feedback"):
                df = pd.DataFrame(st.session_state.feedback_log)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"feedback_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Main content area
    if not gemini_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to start using the Math Tutor.")
        st.info("""
        **To get started:**
        1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Start asking math questions!
        """)
        return
    
    # Initialize or get math agent
    if st.session_state.math_agent is None:
        with st.spinner("🔄 Initializing Math Tutor... This may take a moment."):
            st.session_state.math_agent = get_math_agent(gemini_key, search_key)
    
    if st.session_state.math_agent is None:
        st.error("❌ Failed to initialize the Math Tutor. Please check your API key and try again.")
        return
    
    st.success("✅ Math Tutor is ready! Ask me any math question.")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        display_chat_message(message["role"], message["content"])
        
        # Add feedback buttons for assistant messages
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if st.button("👍", key=f"helpful_{i}", help="This answer was helpful"):
                    if i > 0:  # Ensure there's a previous user message
                        user_msg = st.session_state.messages[i-1]["content"]
                        save_feedback_to_session(user_msg, message["content"], "helpful")
                        st.success("Thank you for your feedback! 👍")
            
            with col2:
                if st.button("👎", key=f"not_helpful_{i}", help="This answer was not helpful"):
                    if i > 0:  # Ensure there's a previous user message
                        user_msg = st.session_state.messages[i-1]["content"]
                        save_feedback_to_session(user_msg, message["content"], "not_helpful")
                        st.info("Thank you for your feedback! We'll improve. 👎")

    # Chat input
    if prompt := st.chat_input("💬 Ask your math question here... (e.g., 'What is 15% of 200?')"):
        # Display user message
        display_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking... Let me solve this for you."):
                try:
                    result = st.session_state.math_agent.get_answer(prompt)
                    response = result["answer"]
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show additional info if available
                    if result.get("context_used"):
                        with st.expander("📚 Context Used", expanded=False):
                            st.text(result.get("context", "Context information not available"))
                    
                    # Add success/error indicators
                    if result["success"]:
                        st.success("✅ Answer generated successfully")
                    else:
                        st.warning("⚠️ Answer may need refinement")
                
                except Exception as e:
                    response = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(response)
            
            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Math Tutor AI - Powered by Gemini AI and GSM8K Dataset<br>
        For educational purposes only. Always verify important calculations.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

