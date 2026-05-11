# =========================================================
# IntelliLearn-AI
# Modular AI Study Assistant with RAG, ML and Evaluation
# =========================================================

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import os
import re
import json
import random
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from groq import Groq

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from pymongo import MongoClient

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", None)
)

MONGO_URI = (
    os.getenv("MONGO_URI")
)

# =========================================================
# SAFETY CHECKS
# =========================================================
if not GROQ_API_KEY:

    st.error(
        "❌ GROQ_API_KEY not found."
    )

    st.stop()

# =========================================================
# CLIENTS
# =========================================================
client = Groq(
    api_key=GROQ_API_KEY
)

collection = None

if MONGO_URI:

    try:

        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000
        )

        mongo_client.server_info()

        db = mongo_client[
            "intellilearn_ai"
        ]

        collection = db[
            "reports"
        ]

    except Exception:

        collection = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(

    page_title=
    "IntelliLearn-AI",

    page_icon=
    "🧠",

    layout=
    "wide"
)

# =========================================================
# APP HEADER
# =========================================================
st.title(
    "🧠 IntelliLearn-AI"
)

st.caption(
    """
Modular AI Study Assistant with
RAG, ML and Evaluation System
"""
)

# =========================================================
# SESSION INIT
# =========================================================
def init_session():

    defaults = {

        "chat_history": [],

        "vector_store": None,

        "last_output": "",

        "mcq_json": None,

        "subjective_question": None,

        "user_stats": {},

        "model_results": None,

        "best_model": None,

        "ml_trained": False
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


init_session()

# =========================================================
# LOGIN
# =========================================================
if "user_id" not in st.session_state:

    st.session_state.user_id = None

if not st.session_state.user_id:

    st.subheader(
        "🔐 Login"
    )

    username = st.text_input(
        "Enter Username"
    )

    if st.button(
        "Login"
    ):

        username = (
            username.strip()
            if username else ""
        )

        if username:

            st.session_state.user_id = username

            st.success(
                f"Welcome {username}"
            )

            st.rerun()

        else:

            st.warning(
                "Please enter username."
            )

    st.stop()

# =========================================================
# SAFETY FUNCTIONS
# =========================================================
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

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header(
        "⚙️ Control Panel"
    )

    st.write(
        f"👤 {st.session_state.user_id}"
    )

    if st.button(
        "Logout"
    ):

        st.session_state.user_id = None

        st.rerun()

    st.markdown("---")

    # =====================================================
    # PDF UPLOAD
    # =====================================================
    st.subheader(
        "📄 Upload PDFs"
    )

    files = st.file_uploader(

        "Upload Study PDFs",

        type=["pdf"],

        accept_multiple_files=True
    )

    # =====================================================
    # RAG CONTROLS
    # =====================================================
    st.subheader(
        "🧠 RAG Controls"
    )

    chunk_size = st.slider(

        "Chunk Size",

        200,
        2000,
        500,
        100
    )

    chunk_overlap = st.slider(

        "Chunk Overlap",

        0,
        500,
        100,
        10
    )

    top_k = st.slider(

        "Top-K Retrieval",

        1,
        10,
        5
    )

    temperature = st.slider(

        "Temperature",

        0.0,
        1.0,
        0.4,
        0.1
    )

    use_memory = st.checkbox(
        "Use Memory",
        value=True
    )

    st.markdown("---")

    # =====================================================
    # MODES
    # =====================================================
    study_mode = st.selectbox(

        "Select Mode",

        [

            "Ask Question",

            "Summarize",

            "Important Points",

            "Notes",

            "Explain Simple",

            "General Chatbot",

            "MCQ Test",

            "Subjective Evaluation",

            "Dataset Analysis",

            "ML Prediction",

            "Analytics Dashboard",

            "Time Series Demo"
        ]
    )

    st.markdown("---")

    if st.button(
        "Clear Chat"
    ):

        st.session_state.chat_history = []

        st.rerun()

    if st.button(
        "Reset PDFs"
    ):

        st.session_state.vector_store = None

        st.success(
            "PDF memory cleared."
        )

# =========================================================
# PROCESS PDF
# =========================================================
if (
    files
    and
    st.session_state.vector_store is None
):

    documents = []

    for file in files:

        pdf = PdfReader(file)

        for i, page in enumerate(
            pdf.pages
        ):

            text = page.extract_text()

            if text:

                documents.append({

                    "text": text,

                    "page": i + 1
                })

    splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap
        )
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

                "page":
                doc["page"]
            })

    embeddings = (
        HuggingFaceEmbeddings(

            model_name=
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    st.session_state.vector_store = (
        FAISS.from_texts(

            chunks,

            embeddings,

            metadatas=metadatas
        )
    )

    st.success(
        "✅ PDFs Processed Successfully"
    )

# =========================================================
# QUICK ACTIONS
# =========================================================
st.subheader(
    "⚡ Quick Actions"
)

c1, c2, c3 = st.columns(3)

if c1.button("Summarize"):

    study_mode = "Summarize"

if c2.button("Generate MCQ"):

    study_mode = "MCQ Test"

if c3.button("Explain Simple"):

    study_mode = "Explain Simple"

# =========================================================
# GENERAL CHATBOT
# =========================================================
if study_mode == "General Chatbot":

    st.header(
        "💬 General AI Chatbot"
    )

    query = st.chat_input(
        "Ask anything..."
    )

    if query:

        st.session_state.chat_history.append(
            ("user", query)
        )

        with st.spinner(
            "Thinking..."
        ):

            response = (
                client.chat.completions.create(

                    model=
                    "llama-3.1-8b-instant",

                    temperature=
                    temperature,

                    messages=[

                        {

                            "role": "system",

                            "content":
                            "You are a helpful AI assistant."
                        },

                        {

                            "role": "user",

                            "content":
                            query
                        }
                    ]
                )
            )

        answer = (
            response
            .choices[0]
            .message.content
        )

        st.session_state.chat_history.append(
            ("bot", answer)
        )

        st.markdown(answer)

# =========================================================
# RAG SYSTEM
# =========================================================
elif study_mode in [

    "Ask Question",

    "Summarize",

    "Important Points",

    "Notes",

    "Explain Simple"
]:

    st.header(
        f"📘 {study_mode}"
    )

    query = None

    if study_mode == "Ask Question":

        query = st.chat_input(
            "Ask question from PDFs..."
        )

    else:

        if st.button("Run"):

            query = "RUN"

    if query and st.session_state.vector_store:

        if detect_unfair_query(query):

            st.warning(
                savage_reply()
            )

            st.stop()

        docs = (
            st.session_state.vector_store
            .similarity_search(

                query
                if study_mode == "Ask Question"
                else "summary",

                k=top_k
            )
        )

        context = "\n\n".join([

            d.page_content
            for d in docs
        ])

        previous = (

            st.session_state.last_output
            if use_memory
            else ""
        )

        # =================================================
        # PROMPTS
        # =================================================
        if study_mode == "Summarize":

            prompt = f"""
Give structured summary.

{context}
"""

        elif study_mode == "Important Points":

            prompt = f"""
Extract exam important points.

{context}
"""

        elif study_mode == "Notes":

            prompt = f"""
Create bullet-point notes.

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
{previous}

Document Context:
{context}

Question:
{query}
"""

        with st.spinner(
            "Generating..."
        ):

            response = (
                client.chat.completions.create(

                    model=
                    "llama-3.1-8b-instant",

                    temperature=
                    temperature,

                    messages=[

                        {

                            "role": "user",

                            "content":
                            prompt
                        }
                    ]
                )
            )

        answer = (
            response
            .choices[0]
            .message.content
        )

        st.session_state.last_output = answer

        st.session_state.chat_history.append(
            ("bot", answer)
        )

        st.markdown(answer)

        st.download_button(

            "📥 Download Output",

            answer,

            file_name="output.txt"
        )

        with st.expander(
            "📄 Sources"
        ):

            for d in docs[:3]:

                st.write(
                    f"📍 Page {d.metadata.get('page')}"
                )

                st.write(
                    d.page_content[:300]
                )

# =========================================================
# MCQ TEST
# =========================================================
elif study_mode == "MCQ Test":

    st.header(
        "📝 AI MCQ Test"
    )

    if st.button(
        "Generate MCQs"
    ):

        if not st.session_state.vector_store:

            st.warning(
                "Upload PDFs first."
            )

        else:

            docs = (
                st.session_state.vector_store
                .similarity_search(

                    "important concepts",

                    k=top_k
                )
            )

            context = "\n\n".join([

                d.page_content
                for d in docs
            ])

            prompt = f"""
Generate EXACTLY 5 MCQs.

Return valid JSON only.

Context:
{context}
"""

            response = (
                client.chat.completions.create(

                    model=
                    "llama-3.1-8b-instant",

                    temperature=0.3,

                    messages=[

                        {

                            "role": "user",

                            "content":
                            prompt
                        }
                    ]
                )
            )

            raw = (
                response
                .choices[0]
                .message.content
            )

            try:

                json_match = re.search(
                    r'\[.*\]',
                    raw,
                    re.DOTALL
                )

                if json_match:

                    parsed = json.loads(
                        json_match.group()
                    )

                    st.session_state.mcq_json = parsed

                    st.success(
                        "MCQs generated."
                    )

                else:

                    st.error(
                        "JSON parsing failed."
                    )

            except Exception:

                st.error(
                    "MCQ generation failed."
                )

    # =====================================================
    # DISPLAY MCQS
    # =====================================================
    if st.session_state.mcq_json:

        score = 0

        for i, q in enumerate(

            st.session_state.mcq_json
        ):

            st.subheader(
                f"Q{i+1}"
            )

            st.write(
                q["question"]
            )

            answer = st.radio(

                "Choose",

                ["A", "B", "C", "D"],

                key=f"mcq_{i}"
            )

        if st.button(
            "Submit MCQ"
        ):

            score = 0

            total = len(
                st.session_state.mcq_json
            )

            for i, q in enumerate(
                st.session_state.mcq_json
            ):

                if (
                    st.session_state.get(
                        f"mcq_{i}"
                    )
                    ==
                    q["answer"]
                ):

                    score += 1

            user = st.session_state.user_id

            if user not in st.session_state.user_stats:

                st.session_state.user_stats[user] = {

                    "attempts": 0,

                    "best_score": 0,

                    "last_score": 0
                }

            stats = (
                st.session_state.user_stats[user]
            )

            stats["attempts"] += 1

            stats["last_score"] = score

            if score > stats["best_score"]:

                stats["best_score"] = score

            st.success(
                f"Score: {score}/{total}"
            )

# =========================================================
# SUBJECTIVE EVALUATION
# =========================================================
elif study_mode == "Subjective Evaluation":

    st.header(
        "📄 Subjective Evaluation"
    )

    if st.button(
        "Generate Question"
    ):

        if not st.session_state.vector_store:

            st.warning(
                "Upload PDFs first."
            )

        else:

            docs = (
                st.session_state.vector_store
                .similarity_search(

                    "important concepts",

                    k=top_k
                )
            )

            context = "\n\n".join([

                d.page_content
                for d in docs
            ])

            prompt = f"""
Generate ONE descriptive exam question.

Context:
{context}
"""

            response = (
                client.chat.completions.create(

                    model=
                    "llama-3.1-8b-instant",

                    messages=[

                        {

                            "role": "user",

                            "content":
                            prompt
                        }
                    ]
                )
            )

            st.session_state.subjective_question = (

                response
                .choices[0]
                .message.content
            )

    if st.session_state.subjective_question:

        st.subheader(
            "Generated Question"
        )

        st.write(
            st.session_state.subjective_question
        )

        student_answer = st.text_area(
            "Write your answer"
        )

        if st.button(
            "Evaluate"
        ):

            prompt = f"""
Evaluate this answer.

Question:
{st.session_state.subjective_question}

Answer:
{student_answer}

Return:
- Score out of 10
- Feedback
"""

            response = (
                client.chat.completions.create(

                    model=
                    "llama-3.1-8b-instant",

                    messages=[

                        {

                            "role": "user",

                            "content":
                            prompt
                        }
                    ]
                )
            )

            evaluation = (

                response
                .choices[0]
                .message.content
            )

            st.markdown(
                evaluation
            )

# =========================================================
# DATASET ANALYSIS
# =========================================================
elif study_mode == "Dataset Analysis":

    st.header(
        "📊 Dataset Analysis"
    )

    try:

        df = pd.read_csv(
            "data/student_performance.csv"
        )

        st.dataframe(df.head())

        st.subheader(
            "Dataset Shape"
        )

        st.write(df.shape)

        st.subheader(
            "Missing Values"
        )

        st.write(
            df.isnull().sum()
        )

        st.subheader(
            "Statistics"
        )

        st.write(
            df.describe()
        )

        numeric_cols = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        selected_col = st.selectbox(

            "Select Column",

            numeric_cols
        )

        fig, ax = plt.subplots()

        df[selected_col].hist(ax=ax)

        ax.set_title(
            selected_col
        )

        st.pyplot(fig)

        st.subheader(
            "Correlation Matrix"
        )

        st.write(
            df.corr(numeric_only=True)
        )

    except Exception as e:

        st.error(str(e))

# =========================================================
# ML PREDICTION
# =========================================================
elif study_mode == "ML Prediction":

    st.header(
        "🤖 ML Prediction System"
    )

    try:

        df = pd.read_csv(
            "data/student_performance.csv"
        )

        # =================================================
        # CREATE LABELS
        # =================================================
        def classify(score):

            if score >= 80:

                return "Excellent"

            elif score >= 60:

                return "Average"

            return "Poor"

        df["performance"] = (
            df["exam_score"]
            .apply(classify)
        )

        # =================================================
        # DROP ID
        # =================================================
        df = df.drop(
            columns=["student_id"]
        )

        X = df.drop(
            columns=[
                "exam_score",
                "performance"
            ]
        )

        y = df["performance"]

        categorical_cols = (
            X.select_dtypes(
                include=["object"]
            ).columns
        )

        numeric_cols = (
            X.select_dtypes(
                exclude=["object"]
            ).columns
        )

        numeric_transformer = Pipeline([

            (

                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (

                "scaler",

                StandardScaler()
            )
        ])

        categorical_transformer = Pipeline([

            (

                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (

                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        preprocessor = (
            ColumnTransformer([

                (

                    "num",

                    numeric_transformer,

                    numeric_cols
                ),

                (

                    "cat",

                    categorical_transformer,

                    categorical_cols
                )
            ])
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,
                y,

                test_size=0.2,

                random_state=42
            )
        )

        # =================================================
        # TRAIN MODELS
        # =================================================
        if st.button(
            "Train Models"
        ):

            models = {

                "Logistic Regression":

                LogisticRegression(
                    max_iter=1000
                ),

                "Random Forest":

                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            }

            results = {}

            best_model = None
            best_acc = 0

            for name, model in models.items():

                pipeline = Pipeline([

                    (

                        "preprocessor",

                        preprocessor
                    ),

                    (

                        "model",

                        model
                    )
                ])

                pipeline.fit(
                    X_train,
                    y_train
                )

                preds = pipeline.predict(
                    X_test
                )

                acc = accuracy_score(
                    y_test,
                    preds
                )

                results[name] = {

                    "accuracy": acc,

                    "precision":
                    precision_score(
                        y_test,
                        preds,
                        average="weighted"
                    ),

                    "recall":
                    recall_score(
                        y_test,
                        preds,
                        average="weighted"
                    ),

                    "f1":
                    f1_score(
                        y_test,
                        preds,
                        average="weighted"
                    )
                }

                if acc > best_acc:

                    best_acc = acc

                    best_model = pipeline

            st.session_state.model_results = results

            st.session_state.best_model = best_model

            st.session_state.ml_trained = True

            os.makedirs(
                "models",
                exist_ok=True
            )

            with open(
                "models/best_model.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    best_model,
                    f
                )

            st.success(
                "Models trained successfully."
            )

        # =================================================
        # SHOW RESULTS
        # =================================================
        if st.session_state.model_results:

            st.subheader(
                "📊 Model Comparison"
            )

            st.write(
                pd.DataFrame(
                    st.session_state.model_results
                ).T
            )

        # =================================================
        # USER PREDICTION
        # =================================================
        if st.session_state.ml_trained:

            st.subheader(
                "🎯 Student Prediction"
            )

            user_input = {}

            for col in X.columns:

                if col in numeric_cols:

                    user_input[col] = (
                        st.number_input(
                            col,
                            value=1.0
                        )
                    )

                else:

                    user_input[col] = (
                        st.text_input(col)
                    )

            if st.button(
                "Predict Performance"
            ):

                input_df = pd.DataFrame(
                    [user_input]
                )

                prediction = (
                    st.session_state.best_model
                    .predict(input_df)[0]
                )

                probabilities = (
                    st.session_state.best_model
                    .predict_proba(input_df)[0]
                )

                confidence = (
                    np.max(probabilities)
                    * 100
                )

                st.success(
                    f"""
Prediction:
{prediction}

Confidence:
{confidence:.2f}%
"""
                )

    except Exception as e:

        st.error(str(e))

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================
elif study_mode == "Analytics Dashboard":

    st.header(
        "📈 Analytics Dashboard"
    )

    stats = (
        st.session_state.user_stats.get(

            st.session_state.user_id,

            {

                "attempts": 0,

                "best_score": 0,

                "last_score": 0
            }
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Attempts",
        stats["attempts"]
    )

    c2.metric(
        "Best Score",
        stats["best_score"]
    )

    c3.metric(
        "Last Score",
        stats["last_score"]
    )

# =========================================================
# TIME SERIES DEMO
# =========================================================
elif study_mode == "Time Series Demo":

    st.header(
        "📈 Time Series Demo"
    )

    try:

        df = pd.read_csv(
            "data/student_performance.csv"
        )

        series = (
            df["exam_score"]
            .rolling(window=5)
            .mean()
        )

        fig, ax = plt.subplots()

        ax.plot(series)

        ax.set_title(
            "Moving Average of Exam Scores"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(str(e))

# =========================================================
# CHAT HISTORY
# =========================================================
st.markdown("---")

st.subheader(
    "🕘 Chat History"
)

for role, msg in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"🧑 {msg}"
        )

    else:

        st.markdown(
            f"🤖 {msg}"
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    """
🚀 IntelliLearn-AI
| RAG + ML + Evaluation Platform
"""
)