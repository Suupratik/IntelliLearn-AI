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

# 🌍 Alignment with UN Sustainable Development Goal (SDG 4)

## 🎯 SDG 4 — Quality Education

IntelliLearn-AI directly supports:

> **United Nations Sustainable Development Goal 4 (SDG 4): Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.**

### 📚 Educational Contributions

- AI-assisted personalized learning
- Interactive digital education
- Automated student assessment systems
- Intelligent revision support
- AI-powered educational accessibility
- Self-learning enhancement using Generative AI

The project demonstrates how Artificial Intelligence can be responsibly applied to modern educational ecosystems.

---

# ✨ Key Features

---

# 📄 1. RAG-Based Document Intelligence

### Features:
- Upload multiple PDFs
- Extract multi-page text
- Configurable chunking
- Semantic embeddings using HuggingFace
- FAISS vector storage
- Top-K similarity retrieval
- Context-aware AI answers
- Source/page-level retrieval transparency

### Pipeline:
PDF → Chunking → Embeddings → FAISS → Retrieval → LLM Response

### Technologies:
- LangChain
- FAISS
- Sentence Transformers
- Groq LLaMA

---

# 💬 2. General AI Chatbot

### Features:
- Conversational AI assistant
- Adjustable temperature
- Educational assistance
- Contextual interaction
- Independent of uploaded PDFs
- Conversational memory support

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

# 📝 4. AI MCQ Evaluation System

### Features:
- Context-aware MCQ generation
- Questions generated directly from PDFs
- Multiple-choice evaluation
- Automatic scoring
- Correct/wrong answer analysis
- Explanation generation
- Percentage calculation
- Student attempt tracking
- Downloadable MCQ reports
- MongoDB score persistence

---

# 📄 5. Subjective Answer Evaluation

### Features:
- AI-generated descriptive questions
- Student answer submission
- Automated evaluation
- AI-generated feedback
- Score generation
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
- Model persistence using Pickle

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
- Extracurricular participation
- etc.

### ML Workflow:
Dataset → Preprocessing → Training → Evaluation → Prediction

---

# 📊 7. Exploratory Data Analysis (EDA)

### Features:
- Statistical summaries
- Missing value analysis
- Correlation analysis
- Dataset visualization
- Histogram generation
- Numeric feature insights

### Libraries:
- Pandas
- NumPy
- Matplotlib

---

# 📈 8. Analytics Dashboard

### Tracks:
- Student attempts
- Best score
- Last score
- Database report count
- AI interaction statistics
- Learning activity tracking
- Historical performance tracking
- Recent learning history
- AI-generated learning insights
- MongoDB-powered educational memory

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
- Historical evaluation storage
- Personalized learning memory
- AI-assisted educational recommendations
- Performance trend tracking

---

# 🛡️ 10. Smart Educational Guardrails

Detects unethical queries such as:
- “paper leak”
- “predict exact question”

and prevents misuse of the system.

---

# 🔁 11. Conversational Memory System

### Features:
- Stores previous AI outputs
- Enables contextual follow-up questions
- Improves conversational continuity
- Enhances educational interaction quality

---

# 🧠 12. AI Learning Analytics & Memory

### Features:
- MongoDB-powered learning history
- Personalized educational context
- Historical score tracking
- Recent activity retrieval
- AI-assisted learning recommendations
- Educational interaction continuity

### Purpose:
Transforms the system from a temporary chatbot into a persistent AI-powered educational ecosystem.

---

# 📄 13. Export & Reporting System

### Features:
- Download AI-generated notes
- Export evaluation reports
- Download MCQ performance reports
- Save generated educational content

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
| LLM API | GROQ API ⚡ |
| LLM Model | LLaMA 3.1 🧠 |
| Embeddings | Sentence Transformers 🤖 |
| Vector Database | FAISS 📊 |
| PDF Processing | PyPDF2 📄 |
| ML Framework | Scikit-learn 🤖 |
| Database | MongoDB ☁️ |
| Data Processing | Pandas + NumPy 📈 |
| Visualization | Matplotlib 📊 |
| Environment Management | Python-dotenv ⚙️ |

---

# 📂 Project Structure

```text
IntelliLearn-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── data/
│   └── student_habits_performance.csv
│
├── models/
│
├── reports/
│
├── uploads/
│
├── exports/
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

#### Linux / Mac
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file:

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
- Smart revision platform
- Student performance prediction
- Educational analytics system
- AI-assisted self-learning

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
✅ Context-aware AI document interaction  
✅ Automated educational evaluation system  
✅ MongoDB-powered persistent analytics  
✅ Personalized AI learning memory  
✅ Interactive AI learning workflow  
✅ Industry-style capstone architecture  
✅ Real-world AI application deployment  
✅ SDG 4 educational impact alignment  

---

# 🏆 Internship Curriculum Coverage

This project demonstrates practical implementation of:

- Machine Learning Pipelines
- NLP & Retrieval-Augmented Generation (RAG)
- Conversational AI
- Educational Analytics
- MongoDB Database Integration
- Exploratory Data Analysis (EDA)
- AI-based Evaluation Systems
- Model Training & Comparison
- Streamlit AI Deployment
- Real-world AI System Design

The project aligns strongly with industry-oriented AI/ML internship objectives and capstone-level educational AI development.

---

# 🔮 Future Enhancements

- 📌 PDF answer highlighting
- 📱 Mobile optimization
- 📊 Advanced analytics dashboard
- 🌐 Multilingual support
- 🧠 Personalized learning recommendations
- 📄 Multi-document comparison
- 🔍 Confidence scoring
- 🧾 PDF-based citation system

---

# 👨‍💻 Author

## Supratik Mitra  
B.Tech CSE (AI & ML)

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