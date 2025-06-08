# 🧮 Math Tutor AI

An interactive math tutor chatbot powered by the **Gemini API** and **GSM8K dataset**, designed to solve math questions and provide helpful explanations. Built using **Streamlit** for the front-end and a custom **Math Agent** backend.

---

##  Features

- 💬 Ask any math question in natural language
- 🤖 AI-powered answers using Gemini Pro and few-shot reasoning
- 🧠 Context-aware reasoning from the GSM8K dataset
- 👍👎 Feedback collection for continuous improvement
- 📄 CSV download of all feedback logs
- 🌐 Deployed via ngrok (for local + public access)

---

## 🚀 Demo

https://53a6-34-173-34-145.ngrok-free.app/

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM API:** Gemini Pro (`Google Generative AI`)  
- **Data Source:** GSM8K (math reasoning dataset)  
- **Utilities:** Pandas, pyngrok

---

## 🔧 Setup Instructions

### 1. Clone the repository

git clone https://github.com/your-username/math-tutor-ai.git

cd math-tutor-ai

2. Install dependencies
pip install -r requirements.txt

3. Set your API key
Open app.py and insert your Gemini API key:
GEMINI_API_KEY = "your-google-api-key"
TAVILY_API_KEY = None  # Optional, leave as None if not used

4. Run the app
streamlit run app.py
5. (Optional) Expose with ngrok
For Colab or remote sharing:
from pyngrok import ngrok
ngrok.connect(8501)
Use the generated public URL to access the app.
--
