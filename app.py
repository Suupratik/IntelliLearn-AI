# =======================
# IMPORTS
# =======================
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import random
import re
import json
import pickle

from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pymongo import MongoClient

# =======================
# LOAD ENV
# =======================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# =======================
# API CLIENT
# =======================
client = Groq(api_key=GROQ_API_KEY)

# =======================
# MONGODB
# =======================
collection = None

if MONGO_URI:

    try:

        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000
        )

        mongo_client.server_info()

        db = mongo_client["intellilearn_ai"]

        collection = db["student_reports"]

    except Exception:
        collection = None

# =======================
# SAVE REPORT
# =======================
def save_report(data):

    try:

        if collection is not None:
            collection.insert_one(data)

    except Exception:
        pass

# =======================
# UI
# =======================
st.set_page_config(
    page_title="IntelliLearn-AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 IntelliLearn-AI")
st.caption(
    "Modular AI Study Assistant with RAG, ML and Evaluation System"
)

# =======================
# SESSION
# =======================
defaults = {

    "chat_history": [],
    "vector_store": None,
    "last_output": "",
    "last_score": 0,
    "best_score": 0,
    "attempts": 0,
    "mcqs": None,
    "subjective_question": "",
    "ml_trained": False
}

for k, v in defaults.items():

    if k not in st.session_state:
        st.session_state[k] = v

# =======================
# LOGIN
# =======================
if "user_id" not in st.session_state:

    st.session_state.user_id = None

if not st.session_state.user_id:

    st.subheader("🔐 Login")

    username = st.text_input("Enter Username")

    if st.button("Login"):

        if username.strip():

            st.session_state.user_id = username.strip()
            st.rerun()

    st.stop()

# =======================
# SAFETY
# =======================
def detect_unfair_query(query):

    keywords = [
        "paper leak",
        "exact question",
        "predict exact"
    ]

    return any(
        k in query.lower()
        for k in keywords
    )

def savage_reply():

    return random.choice([
        "Go study 😄",
        "No shortcuts 😂",
        "Focus on concepts 😉"
    ])

# =======================
# SIDEBAR
# =======================
with st.sidebar:

    # =======================
    # USER INFO
    # =======================
    st.title("🧠 IntelliLearn-AI")

    st.success(
        f"👤 {st.session_state.user_id}"
    )

    # =======================
    # LOGOUT
    # =======================
    if st.button("🚪 Logout"):

        st.session_state.user_id = None

        st.rerun()

    st.markdown("---")

    # =======================
    # PDF UPLOAD
    # =======================
    st.subheader("📄 Upload Study Materials")

    files = st.file_uploader(

        "Upload PDF Documents",

        type="pdf",

        accept_multiple_files=True
    )

    # =======================
    # RAG CONTROLS
    # =======================
    st.subheader("⚙️ RAG Controls")

    k = st.slider(

        "Top-K Retrieval",

        1,

        10,

        5
    )

    temperature = st.slider(

        "LLM Temperature",

        0.0,

        1.0,

        0.7
    )

    use_memory = st.checkbox(

        "Enable Conversational Memory",

        value=True
    )

    # =======================
    # MODES
    # =======================
    st.subheader("🧠 AI Modes")

    study_mode = st.selectbox(

        "Choose Mode",

        [

            "Ask Question",

            "Summarize",

            "Important Points",

            "Notes",

            "MCQ Test",

            "Explain Simple",

            "General Chatbot",

            "Subjective Evaluation",

            "ML Prediction",

            "Analytics Dashboard",

            "Dataset Analysis"
        ]
    )

    st.markdown("---")

    # =======================
    # QUICK ACTIONS
    # =======================
    st.subheader("⚡ Quick Actions")

    if st.button("🧹 Clear Chat"):

        st.session_state.chat_history = []

        st.success("Chat cleared.")

        st.rerun()

    if st.button("🔄 Reset App"):

        for key in list(st.session_state.keys()):

            del st.session_state[key]

        st.rerun()

    if st.button("🗑 Reset PDFs"):

        st.session_state.vector_store = None

        st.success("PDF knowledge base cleared.")

    st.markdown("---")

    # =======================
    # LEARNING SNAPSHOT
    # =======================
    st.subheader("📊 Learning Snapshot")

    st.metric(

        "Chat Messages",

        len(st.session_state.chat_history)
    )

    st.metric(

        "Attempts",

        st.session_state.attempts
    )

    st.metric(

        "Best Score",

        st.session_state.best_score
    )

    st.metric(

        "Last Score",

        st.session_state.last_score
    )

    st.markdown("---")

    # =======================
    # SYSTEM STATUS
    # =======================
    st.subheader("🛠 System Status")

    st.write(

        f"PDF Knowledge Base: {'✅ Ready' if st.session_state.vector_store else '❌ Not Loaded'}"
    )

    st.write(

        f"ML Models: {'✅ Trained' if st.session_state.ml_trained else '❌ Not Trained'}"
    )

    st.write(

        f"Memory Mode: {'✅ Enabled' if use_memory else '❌ Disabled'}"
    )

    st.markdown("---")

    # =======================
    # SDG INFO
    # =======================
    st.info(

        """
🎯 SDG 4: Quality Education

AI-powered educational assistance for interactive and intelligent learning.
"""
    )
    
# =======================
# PDF PROCESSING
# =======================
if files and st.session_state.vector_store is None:

    documents = []

    for file in files:

        pdf = PdfReader(file)

        for i, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:

                documents.append({

                    "text": text,
                    "page": i + 1
                })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    metadatas = []

    for doc in documents:

        split_texts = splitter.split_text(
            doc["text"]
        )

        for chunk in split_texts:

            chunks.append(chunk)

            metadatas.append({

                "page": doc["page"]
            })

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(
        chunks,
        embeddings,
        metadatas=metadatas
    )

    st.success("✅ PDFs Processed Successfully")

    st.info(
        f"📄 Documents Loaded: {len(documents)}"
    )

    st.info(
        f"🧩 Chunks Created: {len(chunks)}"
    )

# =======================
# DATASET ANALYSIS
# =======================
if study_mode == "Dataset Analysis":

    st.header("📊 Dataset Analysis")

    try:

        df = pd.read_csv(
            "data/student_habits_performance.csv"
        )

        st.dataframe(df.head())

        st.subheader("Missing Values")

        st.write(df.isnull().sum())

        st.subheader("Statistics")

        st.write(df.describe())

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        col = st.selectbox(
            "Select Column",
            numeric_cols
        )

        fig, ax = plt.subplots()

        df[col].hist(ax=ax)

        ax.set_title(col)

        st.pyplot(fig)

    except Exception as e:

        st.error(f"Error: {e}")
        
# =======================
# ML PREDICTION
# =======================
elif study_mode == "ML Prediction":

    st.header("🤖 ML Prediction System")

    try:

        df = pd.read_csv(
            "data/student_habits_performance.csv"
        )

        df = df.drop(columns=["student_id"])

        categorical_cols = df.select_dtypes(
            include="object"
        ).columns

        for col in categorical_cols:

            df[col] = df[col].astype("category").cat.codes

        df["performance"] = (
            df["exam_score"] >= 60
        ).astype(int)

        X = df.drop(
            columns=["exam_score", "performance"]
        )

        y = df["performance"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        if st.button("Train Models"):

            rf_model = RandomForestClassifier()
            rf_model.fit(X_train, y_train)

            rf_preds = rf_model.predict(X_test)

            rf_acc = accuracy_score(
                y_test,
                rf_preds
            )

            lr_model = LogisticRegression(
                max_iter=1000
            )

            lr_model.fit(X_train, y_train)

            lr_preds = lr_model.predict(X_test)

            lr_acc = accuracy_score(
                y_test,
                lr_preds
            )

            if rf_acc >= lr_acc:

                best_model = rf_model
                best_name = "Random Forest"
                best_acc = rf_acc

            else:

                best_model = lr_model
                best_name = "Logistic Regression"
                best_acc = lr_acc

            os.makedirs("models", exist_ok=True)

            pickle.dump(
                best_model,
                open("models/best_model.pkl", "wb")
            )

            st.session_state.ml_trained = True

            st.success(
                "Models trained successfully."
            )

            st.subheader("📊 Model Comparison")

            st.write(
                f"Random Forest Accuracy: {rf_acc:.2f}"
            )

            st.write(
                f"Logistic Regression Accuracy: {lr_acc:.2f}"
            )

            st.success(
                f"Best Model: {best_name}"
            )

        if os.path.exists("models/best_model.pkl"):

            model = pickle.load(
                open("models/best_model.pkl", "rb")
            )

            st.subheader("🎯 Student Prediction")

            user_input = {}

            for col in X.columns:

                if str(df[col].dtype) in ["int64", "float64"]:

                    user_input[col] = st.number_input(
                        col,
                        value=1.0
                    )

                else:

                    user_input[col] = st.number_input(
                        col,
                        value=1.0
                    )

            if st.button("Predict Performance"):

                input_df = pd.DataFrame(
                    [user_input]
                )

                prediction = model.predict(input_df)[0]

                probability = max(
                    model.predict_proba(input_df)[0]
                )

                label = (
                    "Good Performance"
                    if prediction == 1
                    else "Needs Improvement"
                )

                confidence = probability * 100

                st.success(

                    f"""
Prediction:
{label}

Confidence:
{confidence:.2f}%
"""
                )

                st.progress(
                    int(confidence)
                )

    except Exception as e:

        st.error(f"ML Error: {e}")
# =======================
# SUBJECTIVE EVALUATION
# =======================
elif study_mode == "Subjective Evaluation":

    st.header("📝 Subjective Evaluation")

    if st.button("Generate Descriptive Question"):

        if not st.session_state.vector_store:

            st.warning("Upload PDFs first")

        else:

            docs = st.session_state.vector_store.similarity_search(
                "important concepts",
                k=max(k, 5)
            )

            context = "\n\n".join(
                [d.page_content for d in docs]
            )

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[{

                    "role": "user",

                    "content":

                    f"""
Generate ONE university-level descriptive question.

Keep:
- concise
- educational
- exam-oriented

Context:
{context}
"""
                }]
            )

            question = response.choices[0].message.content

            st.session_state.subjective_question = question

    if st.session_state.subjective_question:

        st.subheader("Generated Question")

        st.write(
            st.session_state.subjective_question
        )

        student_answer = st.text_area(
            "Write Your Answer"
        )

        if st.button("Evaluate Answer"):

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[{

                    "role": "user",

                    "content":

                    f"""
Evaluate this student answer.

Question:
{st.session_state.subjective_question}

Student Answer:
{student_answer}

Return:
- Score out of 10
- Feedback
"""
                }]
            )

            evaluation = (
                response.choices[0]
                .message.content
            )

            st.markdown(evaluation)

            report_data = {

                "username":
                st.session_state.user_id,

                "module":
                "Subjective Evaluation",

                "question":
                st.session_state.subjective_question,

                "answer":
                student_answer,

                "evaluation":
                evaluation,

                "timestamp":
                str(datetime.now())
            }

            save_report(report_data)

            report_text = f"""

IntelliLearn-AI Evaluation Report

Student:
{st.session_state.user_id}

Question:
{st.session_state.subjective_question}

Answer:
{student_answer}

Evaluation:
{evaluation}
"""

            st.download_button(

                "📥 Download Report",

                report_text,

                file_name="evaluation_report.txt"
            )
# =======================
# GENERAL CHATBOT
# =======================
elif study_mode == "General Chatbot":

    query = st.chat_input("Ask anything...")

    if query:

        st.session_state.chat_history.append(
            ("user", query)
        )

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            temperature=temperature,

            messages=[

                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },

                {
                    "role": "user",
                    "content": query
                }
            ]
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append(
            ("bot", answer)
        )

        st.markdown(answer)
# =======================
# MAIN RAG SYSTEM
# =======================
else:

    query = None

    # =======================
    # INPUT
    # =======================
    if study_mode == "Ask Question":

        query = st.chat_input(
            "Ask anything..."
        )

    else:

        if st.button("Run"):

            query = "RUN"

    # =======================
    # RAG EXECUTION
    # =======================
    if (
        query or
        study_mode != "Ask Question"
    ) and st.session_state.vector_store:

        # =======================
        # SAFETY CHECK
        # =======================
        if study_mode == "Ask Question":

            if detect_unfair_query(query):

                answer = savage_reply()

                st.markdown(answer)

                st.stop()

            st.session_state.chat_history.append(
                ("user", query)
            )

        # =======================
        # RETRIEVAL
        # =======================
        docs = st.session_state.vector_store.similarity_search(

            query if query else "summary",

            k=max(k, 5)
        )

        context = "\n\n".join(
            [d.page_content for d in docs]
        )

        # =======================
        # MEMORY CONTEXT
        # =======================
        extra_context = ""

        # Session Memory
        if use_memory:

            extra_context += (
                st.session_state.last_output
            )

        # MongoDB Learning Memory
        if collection is not None:

            try:

                recent_history = list(

                    collection.find({

                        "username":
                        st.session_state.user_id

                    }).sort(
                        "timestamp",
                        -1
                    ).limit(3)
                )

                if recent_history:

                    extra_context += "\n\nPrevious Learning History:\n"

                    for item in recent_history:

                        module = item.get(
                            "module",
                            ""
                        )

                        score = item.get(
                            "score",
                            ""
                        )

                        extra_context += f"""

Module:
{module}

Previous Score:
{score}
"""

            except Exception:
                pass
        # =======================
        # PROMPTS
        # =======================
        if study_mode == "Summarize":

            prompt = f"""
Summarize clearly.

{context}
"""

        elif study_mode == "Important Points":

            prompt = f"""
Extract important exam points.

{context}
"""

        elif study_mode == "Notes":

            prompt = f"""
Generate structured notes.

{context}
"""

        # =======================
        # MCQ TEST
        # =======================
        elif study_mode == "MCQ Test":

            prompt = f"""
Generate EXACTLY 5 MCQs from the context.

Return STRICT JSON ONLY.

Format:

[
  {{
    "question": "Question",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Short explanation"
  }}
]

Context:
{context}
"""

        elif study_mode == "Explain Simple":

            prompt = f"""
Explain simply for beginners.

{context}
"""

        else:

            prompt = f"""
Previous Context:
{extra_context}

Document Context:
{context}

Question:
{query}
"""

        # =======================
        # LLM CALL
        # =======================
        with st.spinner("Thinking..."):

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                temperature=temperature,

                messages=[{

                    "role": "user",

                    "content": prompt
                }]
            )

        answer = (
            response
            .choices[0]
            .message.content
        )

        # =======================
        # MCQ EVALUATION SYSTEM
        # =======================
        if study_mode == "MCQ Test":

            try:

                json_match = re.search(

                    r'\[\s*{.*}\s*\]',

                    answer,

                    re.DOTALL
                )

                json_text = (
                    json_match.group()
                )

                mcqs = json.loads(
                    json_text
                )

                st.session_state.mcqs = mcqs

                score = 0

                st.subheader(
                    "📝 MCQ Evaluation Test"
                )

                user_answers = {}

                for i, mcq in enumerate(mcqs):

                    st.markdown(

                        f"""
### Q{i+1}. {mcq['question']}
"""
                    )

                    options = mcq["options"]

                    selected = st.radio(

                        "Choose Answer",

                        list(options.keys()),

                        format_func=lambda x:
                        f"{x}. {options[x]}",

                        key=f"mcq_{i}"
                    )

                    user_answers[i] = selected

                # =======================
                # SUBMIT TEST
                # =======================
                if st.button("Submit MCQ Test"):

                    st.subheader(
                        "📊 Evaluation Result"
                    )

                    feedback_text = ""

                    for i, mcq in enumerate(mcqs):

                        correct = mcq["answer"]

                        explanation = (
                            mcq["explanation"]
                        )

                        selected = (
                            user_answers[i]
                        )

                        # ===================
                        # CORRECT
                        # ===================
                        if selected == correct:

                            score += 1

                            st.success(
                                f"Q{i+1}: Correct ✅"
                            )

                        # ===================
                        # WRONG
                        # ===================
                        else:

                            st.error(

                                f"""
Q{i+1}: Wrong ❌

Correct Answer:
{correct}

Explanation:
{explanation}
"""
                            )

                        feedback_text += f"""

Q{i+1}

Selected:
{selected}

Correct:
{correct}

Explanation:
{explanation}

"""

                    # =======================
                    # FINAL SCORE
                    # =======================
                    total = len(mcqs)

                    percentage = (
                        score / total
                    ) * 100

                    st.success(

                        f"""
Final Score:
{score}/{total}

Percentage:
{percentage:.2f}%
"""
                    )

                    st.progress(
                        int(percentage)
                    )

                    # =======================
                    # ANALYTICS
                    # =======================
                    st.session_state.last_score = score

                    st.session_state.attempts += 1

                    if (
                        score >
                        st.session_state.best_score
                    ):

                        st.session_state.best_score = score

                    # =======================
                    # DATABASE SAVE
                    # =======================
                    report_data = {

                        "username":
                        st.session_state.user_id,

                        "module":
                        "MCQ Evaluation",

                        "score":
                        score,

                        "total":
                        total,

                        "percentage":
                        percentage,

                        "timestamp":
                        str(datetime.now())
                    }

                    save_report(report_data)

                    # =======================
                    # DOWNLOAD REPORT
                    # =======================
                    report_text = f"""

IntelliLearn-AI MCQ Report

Student:
{st.session_state.user_id}

Score:
{score}/{total}

Percentage:
{percentage:.2f}%

Detailed Feedback:
{feedback_text}

Generated On:
{datetime.now()}
"""

                    st.download_button(

                        "📥 Download MCQ Report",

                        report_text,

                        file_name="mcq_report.txt"
                    )

            except Exception as e:

                st.error(
                    "MCQ generation failed."
                )

                st.code(str(e))

        # =======================
        # NORMAL OUTPUT
        # =======================
        else:

            st.session_state.last_output = answer

            st.session_state.chat_history.append(
                ("bot", answer)
            )

            st.markdown(answer)

            st.download_button(
                "Download Answer",
                answer
            )

        # =======================
        # SOURCE DISPLAY
        # =======================
        with st.expander("📄 Sources"):

            for d in docs[:3]:

                st.write(
                    f"📍 Page {d.metadata['page']}"
                )

                st.write(
                    d.page_content[:300]
                )
          
# =======================
# ANALYTICS DASHBOARD
# =======================
if study_mode == "Analytics Dashboard":

    st.header("📊 Analytics Dashboard")

    # =======================
    # SESSION ANALYTICS
    # =======================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Attempts",
        st.session_state.attempts
    )

    col2.metric(
        "Last Score",
        st.session_state.last_score
    )

    col3.metric(
        "Best Score",
        st.session_state.best_score
    )

    st.markdown("---")

    # =======================
    # DATABASE ANALYTICS
    # =======================
    if collection is not None:

        try:

            total_reports = (
                collection.count_documents({})
            )

            st.metric(
                "Database Reports",
                total_reports
            )

            st.markdown("---")

            # =======================
            # USER HISTORY
            # =======================
            st.subheader(
                "🕘 Recent Learning History"
            )

            user_reports = list(

                collection.find({

                    "username":
                    st.session_state.user_id

                }).sort(
                    "timestamp",
                    -1
                ).limit(5)
            )

            if not user_reports:

                st.info(
                    "No historical reports found."
                )

            else:

                for report in user_reports:

                    module = report.get(
                        "module",
                        "Unknown"
                    )

                    timestamp = report.get(
                        "timestamp",
                        "N/A"
                    )

                    score = report.get(
                        "score",
                        "N/A"
                    )

                    percentage = report.get(
                        "percentage",
                        "N/A"
                    )

                    st.markdown(

                        f"""
### 📘 {module}

🕒 {timestamp}

🏆 Score: {score}

📊 Percentage: {percentage}
"""
                    )

                    st.markdown("---")

            # =======================
            # AI LEARNING INSIGHT
            # =======================
            st.subheader(
                "🧠 AI Learning Insight"
            )

            if (
                st.session_state.best_score >= 4
            ):

                st.success(

                    """
Strong performance detected.

Recommendation:
- Continue revision practice
- Attempt advanced questions
"""
                )

            elif (
                st.session_state.attempts >= 3
            ):

                st.warning(

                    """
Performance improvement recommended.

Suggested Actions:
- Revise important points
- Practice MCQs again
- Use 'Explain Simple' mode
"""
                )

            else:

                st.info(

                    """
Keep practicing consistently to improve learning analytics.
"""
                )

        except Exception as e:

            st.error(
                f"Analytics error: {e}"
            )

    else:

        st.warning(
            "MongoDB not connected."
        )
  
# =======================
# CHAT HISTORY
# =======================
st.markdown("---")

st.subheader("🕘 Chat History")

for role, msg in st.session_state.chat_history:

    if role == "user":

        st.markdown(f"🧑 {msg}")

    else:

        st.markdown(f"🤖 {msg}")