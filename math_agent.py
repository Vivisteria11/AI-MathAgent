import json
import os
import requests
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime

# Try importing faiss with fallback
try:
    import faiss
except ImportError:
    print("Warning: faiss not found. Install with: pip install faiss-cpu")
    faiss = None

class MathAgent:
    def __init__(self, gemini_api_key, search_api_key=None):
        """Initialize the Math Agent with API keys"""
        self.gemini_api_key = gemini_api_key
        self.search_api_key = search_api_key
        self.model = None
        self.index = None
        self.questions = []
        self.answers = []
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components"""
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            self._load_or_build_vector_db()
        except Exception as e:
            print(f"Error loading components: {e}")
    
    # Guardrails
    def input_guardrail(self, question):
        """Validate input question"""
        disallowed_keywords = ["hack", "cheat", "game", "politics"]
        
        if not question or not question.strip():
            return False, "Question cannot be empty."
        
        if any(word in question.lower() for word in disallowed_keywords):
            return False, "Only math-related educational questions are allowed."
        
        if len(question) > 500:
            return False, "Question is too long. Please keep it under 500 characters."
        
        return True, question.strip()

    def output_guardrail(self, answer):
        """Validate output answer"""
        if not answer or not answer.strip():
            return False, "Sorry, we couldn't generate a valid answer."
        
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

    # Vector DB operations
    def build_and_save_vector_db(self, limit=None):
        """Build vector database from GSM8K dataset"""
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        try:
            print("Loading dataset...")
            if limit:
                dataset = load_dataset("gsm8k", "main", split=f"train[:{limit}]")
            else:
                dataset = load_dataset("gsm8k", "main")["train"]
            
            questions = [item['question'] for item in dataset]
            answers = [item['answer'] for item in dataset]
            
            print("Generating embeddings...")
            embeddings = self.model.encode(questions, show_progress_bar=True).astype('float32')
            
            print("Building FAISS index...")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            
            # Save to files
            faiss.write_index(index, "gsm8k_faiss.index")
            
            with open("questions.json", "w") as f:
                json.dump(questions, f)
            
            with open("answers.json", "w") as f:
                json.dump(answers, f)
            
            print(f"Vector database built with {len(questions)} examples")
            return True
        
        except Exception as e:
            print(f"Error building vector database: {e}")
            return False

    def _load_or_build_vector_db(self):
        """Load existing vector DB or build new one"""
        if faiss is None:
            print("Warning: FAISS not available. Vector search disabled.")
            return
        
        try:
            # Try to load existing files
            if (os.path.exists("gsm8k_faiss.index") and 
                os.path.exists("questions.json") and 
                os.path.exists("answers.json")):
                
                self.index = faiss.read_index("gsm8k_faiss.index")
                
                with open("questions.json", "r") as f:
                    self.questions = json.load(f)
                
                with open("answers.json", "r") as f:
                    self.answers = json.load(f)
                
                print(f"Loaded vector database with {len(self.questions)} examples")
            else:
                print("Vector database not found. Building new one...")
                if self.build_and_save_vector_db(limit=1000):  # Limit for faster setup
                    self._load_or_build_vector_db()  # Recursive call to load
        
        except Exception as e:
            print(f"Error loading vector database: {e}")

    def query_vector_db(self, query, top_k=3):
        """Query vector database for similar questions"""
        if self.index is None or not self.questions:
            return ""
        
        try:
            query_embedding = self.model.encode([query]).astype('float32')
            D, I = index.search(query_embedding, k=top_k)
            
            context_parts = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.questions):
                    context_parts.append(f"Example {i+1}:\nQ: {self.questions[idx]}\nA: {self.answers[idx]}")
            
            return "\n\n".join(context_parts)
        
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return ""

    # LLM integration
    def query_gemini_rag(self, query, context):
        """Query Gemini with RAG context"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
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

    # Web search fallback
    def web_search_solution(self, query):
        """Fallback web search using Tavily API"""
        if not self.search_api_key:
            return "No additional context available. Please try rephrasing your question."
        
        try:
            url = "https://api.tavily.com/search"
            headers = {"Authorization": f"Bearer {self.search_api_key}"}
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

    # Main processing function
    def get_answer(self, question):
        """Main function to get answer for a math question"""
        # Input validation
        valid, filtered_question = self.input_guardrail(question)
        if not valid:
            return {
                "success": False,
                "answer": filtered_question,
                "context_used": False
            }
        
        # Get context from vector DB
        context = self.query_vector_db(filtered_question)
        
        # If no context available, try web search
        if not context.strip():
            if self.search_api_key:
                web_answer = self.web_search_solution(filtered_question)
                return {
                    "success": True,
                    "answer": web_answer,
                    "context_used": False,
                    "source": "web_search"
                }
            else:
                return {
                    "success": False,
                    "answer": "Sorry, I couldn't find similar examples. Please try rephrasing your math question.",
                    "context_used": False
                }
        
        # Generate answer using Gemini with RAG
        answer = self.query_gemini_rag(filtered_question, context)
        
        # Output validation
        valid_output, safe_answer = self.output_guardrail(answer)
        
        return {
            "success": valid_output,
            "answer": safe_answer,
            "context_used": True,
            "context": context[:200] + "..." if len(context) > 200 else context  # Truncated context for debugging
        }

    # Feedback system
    def save_feedback(self, question, answer, feedback):
        """Save user feedback to file"""
        try:
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'feedback': feedback
            }
            
            # Append to CSV file
            import pandas as pd
            df = pd.DataFrame([feedback_data])
            
            if os.path.exists("feedback_log.csv"):
                df.to_csv("feedback_log.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("feedback_log.csv", mode='w', header=True, index=False)
            
            return True
        
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

# Standalone usage functions (for non-Streamlit use)
def create_math_agent(gemini_api_key, search_api_key=None):
    """Factory function to create a MathAgent instance"""
    return MathAgent(gemini_api_key, search_api_key)

# Legacy function compatibility
def get_final_answer(question, index, questions, answers, model, gemini_key, search_key=None):
    """Legacy function for backward compatibility"""
    agent = MathAgent(gemini_key, search_key)
    agent.index = index
    agent.questions = questions
    agent.answers = answers
    agent.model = model
    
    result = agent.get_answer(question)
    return result["answer"]

# Example usage
if __name__ == "__main__":
    # Example of how to use the agent
    GEMINI_API_KEY = "AIzaSyA-Y1bbM0aofvo_roKegn3Z_37eAw2ZpWc"
    SEARCH_API_KEY = "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"  # Optional
    
    # Create agent
    agent = create_math_agent(GEMINI_API_KEY, SEARCH_API_KEY)
    
    # Ask a question
    question = "What is 25% of 80?"
    result = agent.get_answer(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Used Context: {result['context_used']}")
  

