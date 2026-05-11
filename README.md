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

> Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.

### Educational Contributions

- AI-assisted personalized learning
- Interactive digital education
- Automated student assessment
- Intelligent revision support
- AI-powered educational accessibility
- Self-learning enhancement using Generative AI

---

# ✨ Key Features

---

# 📄 1. RAG-Based Document Intelligence

### Features
- Upload multiple PDFs
- Multi-page text extraction
- Intelligent chunking
- Semantic embeddings
- FAISS vector storage
- Top-K similarity retrieval
- Context-aware AI answers
- Source/page-level transparency

### Pipeline

```text
PDF Upload
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Vector Storage
   ↓
Similarity Retrieval
   ↓
LLM Response
```

---

# 💬 2. General AI Chatbot

### Features
- Conversational AI assistant
- Educational discussions
- Adjustable temperature
- Follow-up questioning
- Conversational memory support

---

# 📚 3. AI Study Assistant

## 📑 Summarization
Generate concise summaries from uploaded PDFs.

## 📌 Important Point Extraction
Extract exam-oriented concepts.

## 📝 Notes Generation
Create structured revision notes.

## 🧒 Explain Simple
Simplify complex concepts for beginners.

---

# 📝 4. AI MCQ Evaluation System

### Features
- AI-generated MCQs
- Context-aware questions
- Automatic evaluation
- Score calculation
- Percentage generation
- Explanation feedback
- Attempt tracking
- Downloadable reports

---

# 📄 5. Subjective Answer Evaluation

### Features
- AI-generated descriptive questions
- Student answer submission
- Automated evaluation
- Feedback generation
- Downloadable evaluation reports

---

# 🤖 6. Machine Learning Prediction System

### Features
- Student performance prediction
- Real-world dataset analysis
- Multiple model training
- Accuracy comparison
- Confidence-based predictions

### Algorithms Used
- Random Forest Classifier
- Logistic Regression

### Prediction Inputs
- Study hours
- Attendance
- Sleep patterns
- Social media usage
- Mental health
- Internet quality
- Exercise frequency
- Diet quality
- Extracurricular participation

---

# 📊 7. Exploratory Data Analysis (EDA)

### Features
- Statistical summaries
- Missing value analysis
- Histogram visualization
- Numeric feature analysis
- Dataset insights

---

# 📈 8. Analytics Dashboard

### Tracks
- Student attempts
- Best score
- Last score
- Learning history
- Database reports
- AI learning insights
- Educational recommendations

---

# ☁️ 9. MongoDB Integration

### Stores
- MCQ reports
- Evaluation reports
- Analytics history
- Learning records
- User attempts

### Benefits
- Persistent storage
- Historical tracking
- Educational analytics
- Personalized learning memory

---

# 🛡️ 10. Smart Educational Guardrails

Detects unethical queries such as:
- “paper leak”
- “exact question”
- “predict exact”

and prevents misuse of the system.

---

# 🔁 11. Conversational Memory System

### Features
- Context-aware replies
- Previous response memory
- Learning continuity
- Educational interaction enhancement

---

# 📄 12. Export & Reporting System

### Features
- Download AI-generated answers
- Export MCQ reports
- Save evaluation reports
- Generate downloadable educational content

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
Similarity Retrieval
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
| Language | Python |
| UI | Streamlit |
| LLM API | Groq |
| LLM Model | LLaMA 3.1 |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| ML Framework | Scikit-learn |
| Database | MongoDB |
| Visualization | Matplotlib |
| PDF Processing | PyPDF2 |

---

# 📂 Project Structure

```text
IntelliLearn-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   └── student_habits_performance.csv
│
├── models/
├── uploads/
├── reports/
├── exports/
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
- Interactive PDF learning
- Automated revision support
- AI evaluation platform
- Student performance prediction
- Educational analytics system

---

# 🎯 AI/ML Concepts Implemented

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Semantic Search
- Prompt Engineering
- Vector Databases
- Machine Learning Pipelines
- Educational NLP
- Conversational Memory
- Student Analytics

---

# 🔥 Key Highlights

✅ End-to-end AI educational platform  
✅ Combines NLP + ML + RAG in one system  
✅ Automated educational evaluation  
✅ Interactive AI learning workflow  
✅ MongoDB-powered educational analytics  
✅ Personalized learning memory  
✅ Real-world deployment architecture  
✅ Industry-style capstone implementation  
✅ SDG 4 educational alignment  

---

# 🏆 Internship Curriculum Coverage

This project demonstrates implementation of:

- Machine Learning Pipelines
- NLP & Retrieval-Augmented Generation
- Educational Analytics
- Conversational AI
- Streamlit Deployment
- MongoDB Integration
- Exploratory Data Analysis
- AI-based Evaluation Systems
- Model Training & Comparison

The system aligns strongly with industry-oriented AI/ML internship objectives.

---

# 🔮 Future Enhancements

- PDF answer highlighting
- Confidence scoring
- Personalized recommendations
- Multi-document comparison
- Advanced analytics dashboard
- Mobile optimization
- Multilingual support

---

# 👨‍💻 Author

## Supratik Mitra
B.Tech CSE (AI & ML)

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🔗 Share the project
- 🚀 Fork the repository

---

# 🎯 Final Note

IntelliLearn-AI demonstrates how:

> Generative AI + Retrieval Systems + Machine Learning + Educational Analytics

can transform traditional static study material into an intelligent, interactive, and adaptive educational ecosystem.