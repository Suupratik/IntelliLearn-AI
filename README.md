# 🧠 IntelliLearn-AI
## Modular AI Study Assistant with RAG, ML and Evaluation System

---

# 🚀 Overview

IntelliLearn-AI is an advanced AI-powered educational platform that combines:

- 📄 Retrieval-Augmented Generation (RAG)
- 🤖 Machine Learning
- 🧠 Educational NLP
- 📊 Student Analytics
- 📝 Automated Evaluation
- ☁️ MongoDB Persistence

into a single intelligent learning ecosystem.

The system transforms static study materials into an interactive AI-assisted educational platform capable of:

- answering questions from PDFs,
- generating MCQs,
- evaluating descriptive answers,
- predicting student performance,
- and analyzing learning patterns.

---

# ✨ Key Features

---

# 📄 1. RAG-Based Document Intelligence

### Features:
- Upload multiple PDFs
- Extract multi-page text
- Configurable chunk size & overlap
- Semantic embeddings using HuggingFace
- FAISS vector storage
- Top-K similarity retrieval
- Context-aware AI answers
- Source/page-level retrieval transparency

### Pipeline:
PDF → Chunking → Embeddings → FAISS → Retrieval → LLM Response

---

# 💬 2. General AI Chatbot

### Features:
- Conversational AI assistant
- Adjustable temperature
- Educational assistance
- Contextual interaction
- Independent of uploaded PDFs

---

# 📚 3. AI Study Assistant

### Includes:

## 📑 Summarization
Generates structured summaries from uploaded documents.

## 📌 Important Point Extraction
Identifies key exam-oriented concepts.

## 📝 Notes Generation
Creates concise revision-ready notes.

## 🧒 Explain Like Beginner
Simplifies difficult concepts into beginner-friendly explanations.

---

# 📝 4. AI MCQ Generation System

### Features:
- Context-aware MCQ generation
- Questions generated directly from PDFs
- Multiple-choice evaluation
- Automatic scoring
- Student attempt tracking
- MongoDB score persistence

---

# 📄 5. Subjective Answer Evaluation

### Features:
- AI-generated descriptive questions
- Student answer submission
- Automated evaluation
- Feedback generation
- Downloadable evaluation reports
- Persistent database storage

---

# 🤖 6. Machine Learning Prediction System

### Features:
- Student performance prediction
- Real-world dataset analysis
- Multiple ML model training
- Accuracy comparison
- Confidence-based predictions

### Algorithms Used:
- Logistic Regression
- Random Forest Classifier

### Prediction Inputs:
- Study hours
- Attendance
- Sleep patterns
- Social media usage
- Mental health
- Exercise frequency
- Internet quality
- Diet quality
- etc.

---

# 📊 7. Exploratory Data Analysis (EDA)

### Features:
- Statistical summaries
- Missing value analysis
- Correlation analysis
- Dataset visualization
- Histogram generation
- Numeric feature insights

---

# 📈 8. Analytics Dashboard

### Tracks:
- Student attempts
- Best score
- Last score
- Database report count
- AI interaction statistics

---

# ☁️ 9. MongoDB Integration

### Stores:
- MCQ reports
- Evaluation reports
- Student attempts
- User analytics
- Timestamps

### Benefits:
- Persistent storage
- Multi-user tracking
- Educational analytics

---

# 🛡️ 10. Smart Educational Guardrails

Detects unethical queries such as:
- “paper leak”
- “predict exact question”

and prevents misuse of the system.

---

# 🧠 System Architecture

```text
PDF Upload
   ↓
Text Extraction (PyPDF2)
   ↓
Chunking (LangChain Splitter)
   ↓
Embeddings (Sentence Transformers)
   ↓
FAISS Vector Storage
   ↓
User Query
   ↓
Similarity Retrieval (Top-K)
   ↓
Context + Memory Injection
   ↓
LLM Processing (Groq LLaMA)
   ↓
Final AI Response
```

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 🐍 |
| UI | Streamlit 🎨 |
| LLM | GROQ API ⚡ |
| Model | LLaMA 3.1 |
| Embeddings | Sentence Transformers 🤖 |
| Vector Database | FAISS 📊 |
| PDF Processing | PyPDF2 📄 |
| ML Framework | Scikit-learn 🤖 |
| Deep Learning | TensorFlow 🔥 |
| Database | MongoDB ☁️ |
| Data Processing | Pandas + NumPy 📈 |
| Visualization | Matplotlib 📊 |

---

# 📂 Project Structure

```text
IntelliLearn-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   └── student_performance.csv
│
├── models/
│
├── reports/
│
├── uploads/
│
├── assets/
│
└── .streamlit/
    └── config.toml
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd IntelliLearn-AI
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
MONGO_URI=your_mongodb_uri
```

---

## 5️⃣ Run Application

```bash
streamlit run app.py
```

---

# 📚 Educational Applications

- AI-powered study assistant
- Interactive document learning
- Automated test generation
- Smart revision system
- Student performance prediction
- Educational analytics platform

---

# 🎯 AI/ML Concepts Implemented

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Semantic Search
- Prompt Engineering
- Vector Databases
- Embeddings
- Machine Learning Pipelines
- Model Comparison
- Educational NLP
- Conversational Memory
- Student Performance Analytics

---

# 🔥 Key Highlights

✅ End-to-end AI educational platform  
✅ Combines NLP + ML + RAG in one system  
✅ Interactive AI learning workflow  
✅ Persistent educational analytics  
✅ Real-world deployment architecture  
✅ Industry-style capstone project  

---

# 🔮 Future Enhancements

- 📌 PDF answer highlighting
- 📱 Mobile optimization
- 📊 Advanced analytics dashboard
- 🌐 Multilingual support
- 🧠 Personalized learning recommendations
- 📄 Multi-document comparison
- 🔍 Confidence scoring

---

# 👨‍💻 Author

## Supratik Mitra
B.Tech CSE 

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🔗 Share with others
- 🚀 Fork the project

---

# 🎯 Final Note

IntelliLearn-AI demonstrates how:

> Generative AI + Retrieval Systems + Machine Learning + Educational Analytics

can transform traditional static study material into an intelligent, interactive, and adaptive educational ecosystem.