# ------------------------
# Imports and Config
# ------------------------
import streamlit as st
import os
import tempfile
import json
import re 
import random
import time
import logging
import asyncio
import fitz  # PyMuPDF

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Google API key
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCmhSlMWdIkW9SsxMWkLZCAsB8p0trsZyE'

# Configuration
class Settings:
    embed_model = "models/embedding-001"
    llm_model = "gemini-2.0-flash"
    chunk_size = 2000
    chunk_overlap = 200 

settings = Settings()

# ------------------------
# PDF Processing
# ------------------------
def process_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\n', '', text)
        if len(text.strip()) < 100:
            raise ValueError("PDF text extraction failed.")
        return text
    except Exception as e:
        logger.error(f"Primary extraction failed: {str(e)}")
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(file_bytes))
            text = "".join([page.extract_text() + "\n\n" for page in reader.pages])
            return text
        except Exception as fallback_e:
            logger.error(f"Fallback extraction failed: {str(fallback_e)}")
            raise RuntimeError("PDF processing failed.")

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# ------------------------
# RAG Service (Fixed)
# ------------------------
class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None

    async def initialize_qa_system(self, chunks: list[str]):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.embed_model)
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.3)

        prompt = PromptTemplate(
            template="""
            Use the following context to answer the question. If you don't know the answer, 
            just say you don't know. Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA system not initialized.")
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

# ------------------------
# Quiz Service
# ------------------------
class QuizService:
    def __init__(self):
        self.quizzes = {}
        self.llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.7)

    def extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.error(f"Failed to extract JSON from: {text}")
        return None

    def generate_quiz(self, context: str, retries=3) -> dict:
        prompt = f"""
        Generate 10 multiple-choice questions based on the following book content.
        Each question should have 4 options (A, B, C, D) with one correct answer.
        Format your response as a JSON object with this structure:
        {{
            "quiz_id": "unique_id_placeholder",
            "questions": [
                {{
                    "id": 1,
                    "question": "question text",
                    "options": {{
                        "A": "option1",
                        "B": "option2",
                        "C": "option3",
                        "D": "option4"
                    }},
                    "correct": "A"
                }}
            ]
        }}
        Book Content: {context[:10000]}
        """
        for attempt in range(retries):
            try:
                response = self.llm.invoke(prompt)
                quiz_json = self.extract_json(response.content)
                if not quiz_json:
                    raise ValueError("JSON extraction failed")
                quiz_id = f"quiz_{random.randint(1000,9999)}_{int(time.time())}"
                quiz_json["quiz_id"] = quiz_id
                self.quizzes[quiz_id] = quiz_json
                return quiz_json
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(1)
        raise RuntimeError("Failed to generate quiz.")

    def evaluate_quiz(self, quiz_id: str, answers: dict) -> dict:
        if quiz_id not in self.quizzes:
            raise KeyError("Quiz not found")
        quiz = self.quizzes[quiz_id]
        score = 0
        results = []
        for q in quiz["questions"]:
            user_answer = answers.get(str(q["id"]), "").strip().upper()
            correct = q["correct"].strip().upper()
            is_correct = user_answer == correct
            if is_correct: score += 1
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "user_answer": user_answer,
                "correct_answer": correct,
                "is_correct": is_correct,
                "options": q["options"]
            })
        return {
            "score": f"{score}/{len(quiz['questions'])}",
            "detail": results
        }

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="StudyBuddy", layout="wide")
st.title("📚 StudyBuddy")
st.subheader("Learn from books and test your knowledge")

# Initialize session state
for key, default in {
    "book_uploaded": False,
    "book_text": "",
    "chunks": [],
    "quiz_data": None,
    "user_answers": {},
    "quiz_generated": False,
    "quiz_submitted": False,
    "quiz_result": None,
    "rag_service": RAGService(),
    "quiz_service": QuizService()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar Upload
with st.sidebar:
    st.header("Upload Book")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process Book"):
            with st.spinner("Processing..."):
                try:
                    file_bytes = uploaded_file.read()
                    text = process_pdf(file_bytes)
                    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                    st.session_state.book_text = text
                    st.session_state.chunks = chunks
                    # Async RAG initialization
                    asyncio.run(st.session_state.rag_service.initialize_qa_system(chunks))
                    st.session_state.book_uploaded = True
                    st.success("Book processed!")
                    st.info(f"Extracted {len(text)} characters in {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Book processing failed")

# Main Interface
if st.session_state.book_uploaded:
    learn_tab, quiz_tab = st.tabs(["Learning", "Quiz"])
    with learn_tab:
        st.header("Ask Questions")
        question = st.text_input("Ask about the book:", key="question_input")
        if question and st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_service.query(question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Question answering failed")

    with quiz_tab:
        st.header("Test Your Knowledge")
        if not st.session_state.quiz_generated:
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz..."):
                    try:
                        quiz_data = st.session_state.quiz_service.generate_quiz(st.session_state.book_text)
                        st.session_state.quiz_data = quiz_data
                        st.session_state.quiz_generated = True
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.success("Quiz generated!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.exception("Quiz generation failed")

        if st.session_state.quiz_generated and st.session_state.quiz_data:
            quiz = st.session_state.quiz_data
            with st.form("quiz_form"):
                st.subheader("Quiz Questions")
                for q in quiz["questions"]:
                    key = f"q_{q['id']}"
                    if key not in st.session_state.user_answers:
                        st.session_state.user_answers[key] = ""
                    options = q["options"]
                    selected = st.radio(
                        f"{q['question']}",
                        options=list(options.keys()),
                        format_func=lambda k: f"{k}. {options[k]}",
                        key=f"radio_{q['id']}",
                        index=None
                    )
                    st.session_state.user_answers[key] = selected if selected else ""
                submitted = st.form_submit_button("Submit Quiz")
                if submitted:
                    st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            with st.spinner("Evaluating..."):
                try:
                    answers = {
                        key.split("_")[1]: val
                        for key, val in st.session_state.user_answers.items() if key.startswith("q_")
                    }
                    result = st.session_state.quiz_service.evaluate_quiz(
                        st.session_state.quiz_data["quiz_id"], answers
                    )
                    st.session_state.quiz_result = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Quiz evaluation failed")

            if st.session_state.quiz_result:
                result = st.session_state.quiz_result
                st.success(f"## Your Score: {result['score']}")
                with st.expander("Detailed Results"):
                    for res in result["detail"]:
                        status = "✅" if res["is_correct"] else "❌"
                        st.markdown(f"{status} **Q{res['question_id']}:** {res['question']}")
                        st.markdown(f"- Your answer: **{res['user_answer']}**")
                        st.markdown(f"- Correct answer: **{res['correct_answer']}**")
                        for opt, val in res["options"].items():
                            prefix = "✓" if opt == res["correct_answer"] else "-"
                            st.markdown(f"{prefix} **{opt}**: {val}")
                        st.divider()

else:
    st.info("📘 Please upload a book PDF to get started.")
