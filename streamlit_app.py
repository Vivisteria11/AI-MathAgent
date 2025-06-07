import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
import streamlit as st
from vector_db_utils import load_vector_db, get_final_answer, save_feedback
import os

# Set up secrets (for Streamlit Cloud deployment)
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

st.set_page_config(page_title="📘 Math Tutor Agent")
st.title("📘 Math Tutor Agent")
st.write("Ask me any math-related question from GSM8K!")

query = st.text_input("🔍 Ask your question")

if st.button("Submit") and query:
    with st.spinner("Processing your question..."):
        index, questions, answers, model = load_vector_db()
        answer = get_final_answer(query, index, questions, answers, model, GEMINI_KEY, TAVILY_KEY)
        st.markdown(f"### ✅ Answer:\n{answer}")

    feedback = st.radio("💬 Was this helpful?", ("Yes", "No"))
    if st.button("Submit Feedback"):
        save_feedback(query, answer, feedback)
        st.success("✅ Thanks for your feedback!")
