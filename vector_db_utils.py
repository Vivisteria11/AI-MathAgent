import json, os, requests
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Guardrails
def input_guardrail(question):
    disallowed_keywords = ["hack", "cheat", "game", "politics"]
    if not question.strip():
        return False, "Question cannot be empty."
    if any(word in question.lower() for word in disallowed_keywords):
        return False, "Only math-related educational questions are allowed."
    return True, question

def output_guardrail(answer):
    if "I'm not sure" in answer or "hallucination" in answer.lower():
        return False, "Sorry, we couldn't confidently generate an answer."
    return True, answer

# Vector DB creation (run once)
def build_and_save_vector_db():
    dataset = load_dataset("gsm8k", "main")["train"]
    questions = [item['question'] for item in dataset]
    answers = [item['answer'] for item in dataset]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions, show_progress_bar=True).astype('float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "gsm8k_faiss.index")
    json.dump(questions, open("questions.json", "w"))
    json.dump(answers, open("answers.json", "w"))

# Load vector DB
def load_vector_db():
    index = faiss.read_index("gsm8k_faiss.index")
    questions = json.load(open("questions.json"))
    answers = json.load(open("answers.json"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, questions, answers, model

# Query vector DB
def query_vector_db(query, index, questions, answers, model, top_k=3):
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, k=top_k)
    return "\n\n".join([f"Q: {questions[i]}\nA: {answers[i]}" for i in I[0]])

# Gemini LLM
def query_gemini_rag(query, context, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a helpful math tutor. Answer the following question using the context.

### Context:
{context}

### Question:
{query}

### Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Web search fallback
def web_search_solution(query, api_key):
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"query": query, "include_answer": True}
        res = requests.post(url, headers=headers, json=payload)
        return res.json().get("answer", "No relevant result found.")
    except:
        return "Error fetching from web search."

# Final answer routing
def get_final_answer(question, index, questions, answers, model, gemini_key, search_key):
    valid, filtered = input_guardrail(question)
    if not valid:
        return filtered

    context = query_vector_db(filtered, index, questions, answers, model)
    if not context.strip():
        return web_search_solution(filtered, search_key)

    answer = query_gemini_rag(filtered, context, gemini_key)
    valid_output, safe_answer = output_guardrail(answer)
    return safe_answer

# Feedback logger
def save_feedback(question, answer, feedback):
    with open("feedback_log.csv", "a") as f:
        f.write(f"{question},{answer},{feedback}\n")
