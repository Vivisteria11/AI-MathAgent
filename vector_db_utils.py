import json
import os
import requests
import numpy as np
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pandas as pd
from datetime import datetime

# Install faiss-cpu instead of faiss for Streamlit Cloud
try:
    import faiss
except ImportError:
    st.error("Please install faiss-cpu: pip install faiss-cpu")
    st.stop()

# Configuration
@st.cache_resource
def get_model():
    """Load sentence transformer model with caching"""
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Guardrails
def input_guardrail(question):
    """Validate input question"""
    disallowed_keywords = ["hack", "cheat", "game", "politics"]
    
    if not question or not question.strip():
        return False, "Question cannot be empty."
    
    if any(word in question.lower() for word in disallowed_keywords):
        return False, "Only math-related educational questions are allowed."
    
    # Additional length check
    if len(question) > 500:
        return False, "Question is too long. Please keep it under 500 characters."
    
    return True, question.strip()

def output_guardrail(answer):
    """Validate output answer"""
    if not answer or not answer.strip():
        return False, "Sorry, we couldn't generate a valid answer."
    
    # Check for common problematic responses
    problematic_phrases = [
        "I'm not sure", 
        "hallucination", 
        "I cannot", 
        "I don't know",
        "unable to provide"
    ]
    
    if any(phrase in answer.lower() for phrase in problematic_phrases):
        return False, "Sorry, we couldn't confidently generate an answer."
    
    return True, answer.strip()

# Vector DB creation (run once)
@st.cache_data
def build_vector_db():
    """Build vector database with caching"""
    try:
        # Check if files already exist
        if (os.path.exists("gsm8k_faiss.index") and 
            os.path.exists("questions.json") and 
            os.path.exists("answers.json")):
            return True
        
        with st.spinner("Building vector database... This may take a few minutes."):
            dataset = load_dataset("gsm8k", "main", split="train[:1000]")  # Limit for faster loading
            questions = [item['question'] for item in dataset]
            answers = [item['answer'] for item in dataset]
            
            model = get_model()
            embeddings = model.encode(questions, show_progress_bar=False).astype('float32')
            
            # Create FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            
            # Save to files
            faiss.write_index(index, "gsm8k_faiss.index")
            
            with open("questions.json", "w") as f:
                json.dump(questions, f)
            
            with open("answers.json", "w") as f:
                json.dump(answers, f)
            
        return True
    
    except Exception as e:
        st.error(f"Error building vector database: {str(e)}")
        return False

# Load vector DB
@st.cache_resource
def load_vector_db():
    """Load vector database with caching"""
    try:
        if not os.path.exists("gsm8k_faiss.index"):
            if not build_vector_db():
                return None, None, None, None
        
        index = faiss.read_index("gsm8k_faiss.index")
        
        with open("questions.json", "r") as f:
            questions = json.load(f)
        
        with open("answers.json", "r") as f:
            answers = json.load(f)
        
        model = get_model()
        return index, questions, answers, model
    
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None, None, None, None

# Query vector DB
def query_vector_db(query, index, questions, answers, model, top_k=3):
    """Query vector database for similar questions"""
    try:
        query_embedding = model.encode([query]).astype('float32')
        D, I = index.search(query_embedding, k=top_k)
        
        context_parts = []
        for i, idx in enumerate(I[0]):
            if idx < len(questions):  # Safety check
                context_parts.append(f"Example {i+1}:\nQ: {questions[idx]}\nA: {answers[idx]}")
        
        return "\n\n".join(context_parts)
    
    except Exception as e:
        st.error(f"Error querying vector database: {str(e)}")
        return ""

# Gemini LLM
def query_gemini_rag(query, context, api_key):
    """Query Gemini with RAG context"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Use available model
        
        prompt = f"""You are a helpful math tutor. Answer the following question step by step using the provided examples as guidance.

### Examples from similar problems:
{context}

### Question to answer:
{query}

### Instructions:
- Provide a clear, step-by-step solution
- Show your work and reasoning
- Keep the answer focused on math concepts
- If the question is not math-related, politely redirect to math topics

### Answer:"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        
        return response.text.strip() if response.text else "No response generated."
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Web search fallback (optional - remove if you don't have Tavily API)
def web_search_solution(query, api_key):
    """Fallback web search (optional)"""
    if not api_key:
        return "No additional context available. Please try rephrasing your question."
    
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "query": f"math problem: {query}",
            "include_answer": True,
            "max_results": 3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result.get("answer", "No relevant result found.")
    
    except Exception as e:
        return f"Web search unavailable: {str(e)}"

# Main answer function
def get_final_answer(question, index, questions, answers, model, gemini_key, search_key=None):
    """Get final answer with all processing steps"""
    # Input validation
    valid, filtered_question = input_guardrail(question)
    if not valid:
        return filtered_question
    
    # Get context from vector DB
    context = query_vector_db(filtered_question, index, questions, answers, model)
    
    # If no context, try web search (optional)
    if not context.strip() and search_key:
        return web_search_solution(filtered_question, search_key)
    elif not context.strip():
        return "Sorry, I couldn't find similar examples. Please try rephrasing your math question."
    
    # Generate answer using Gemini
    answer = query_gemini_rag(filtered_question, context, gemini_key)
    
    # Output validation
    valid_output, safe_answer = output_guardrail(answer)
    return safe_answer

# Feedback system
def save_feedback(question, answer, feedback):
    """Save user feedback"""
    try:
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'feedback': feedback
        }
        
        # Use session state to store feedback if file writing fails
        if 'feedback_log' not in st.session_state:
            st.session_state.feedback_log = []
        
        st.session_state.feedback_log.append(feedback_data)
        
        # Try to save to file (may not work in some deployments)
        try:
            df = pd.DataFrame([feedback_data])
            if os.path.exists("feedback_log.csv"):
                df.to_csv("feedback_log.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("feedback_log.csv", mode='w', header=True, index=False)
        except:
            pass  # Fail silently if file writing doesn't work
            
        return True
    
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False

