import streamlit as st
import os
import tempfile
import json
import re 
import random
import time
import logging
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Google API key
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"]='AIzaSyCmhSlMWdIkW9SsxMWkLZCAsB8p0trsZyE'
# Configuration
class Settings:
    embed_model = "models/embedding-001"
    llm_model = "gemini-2.0-flash"
    chunk_size = 2000
    chunk_overlap = 200 

settings = Settings()

# PDF Processing with PyMuPDF as primary
def process_pdf(file_bytes: bytes) -> str:
    """Extract and process text from PDF bytes using PyMuPDF"""
    try:
        # Open PDF directly from bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        
        for page in doc:
            text += page.get_text() + "\n\n"
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'-\n', '', text)  # Remove hyphenated line breaks
        
        # Validate we extracted text
        if len(text.strip()) < 100:
            raise ValueError("PDF text extraction failed - document may be scanned or encrypted")
            
        return text
    except Exception as e:
        logger.error(f"PyMuPDF failed: {str(e)}. Trying fallback...")
        # Fallback to simple PDF extraction
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as fallback_e:
            logger.error(f"Fallback extraction failed: {str(fallback_e)}")
            raise RuntimeError("PDF processing failed with all methods")

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# RAG Service (without persistence)
class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embed_model
        )
    
    def initialize_qa_system(self, chunks: list[str]):
        """Initialize vector store and QA system in memory"""
        logger.info("Initializing QA system...")
        # Create in-memory vector store without persistence
        self.vector_store = Chroma.from_texts(
            chunks, 
            self.embeddings
        )
        logger.info("Vector store created")
        
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.3
        )
        
        prompt_template = """
        Use the following context to answer the question. If you don't know the answer, 
        just say you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("QA chain initialized")
    
    def query(self, question: str) -> str:
        """Answer question based on book context"""
        if not self.qa_chain:
            raise ValueError("QA system not initialized")
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

# Quiz Service
class QuizService:
    def __init__(self):
        self.quizzes = {}
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.7
        )
    
    def extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response text"""
        try:
            # Try direct parsing first
            return json.loads(text)
        except json.JSONDecodeError:
            # Handle wrapped JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.error(f"Failed to extract JSON from: {text}")
        return None

    def generate_quiz(self, context: str, retries=3) -> dict:
        """Generate quiz questions from book context"""
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
                }},
                ...
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
                
                # Generate unique quiz ID
                quiz_id = f"quiz_{random.randint(1000,9999)}_{int(time.time())}"
                quiz_json["quiz_id"] = quiz_id
                self.quizzes[quiz_id] = quiz_json
                return quiz_json
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(1)
        
        raise RuntimeError("Failed to generate valid quiz after multiple attempts")

    def evaluate_quiz(self, quiz_id: str, answers: dict) -> dict:
        """Evaluate user's quiz answers"""
        if quiz_id not in self.quizzes:
            raise KeyError("Quiz not found")
        
        quiz = self.quizzes[quiz_id]
        score = 0
        results = []
        
        for q in quiz["questions"]:
            # Get user answer (convert to uppercase)
            user_answer = answers.get(str(q["id"]), "").strip().upper()
            correct_answer = q["correct"].strip().upper()
            is_correct = user_answer == correct_answer
            
            if is_correct:
                score += 1
                
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "options": q["options"]
            })
        
        return {
            "score": f"{score}/{len(quiz['questions'])}",
            "detail": results
        }

# Streamlit App
st.set_page_config(page_title="StudyBuddy", layout="wide")
st.title("📚 StudyBuddy")
st.subheader("Learn from books and test your knowledge")

# Initialize session state
if 'book_uploaded' not in st.session_state:
    st.session_state.book_uploaded = False
if 'book_text' not in st.session_state:
    st.session_state.book_text = ""
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = None
if 'rag_service' not in st.session_state:
    st.session_state.rag_service = RAGService()
if 'quiz_service' not in st.session_state:
    st.session_state.quiz_service = QuizService()

# Sidebar for book upload
with st.sidebar:
    st.header("Upload Book")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Book"):
            with st.spinner("Processing book..."):
                try:
                    file_bytes = uploaded_file.read()
                    text = process_pdf(file_bytes)
                    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                    
                    # Store in session state
                    st.session_state.book_text = text
                    st.session_state.chunks = chunks
                    
                    # Initialize RAG
                    st.session_state.rag_service.initialize_qa_system(chunks)
                    st.session_state.book_uploaded = True
                    st.success("Book processed successfully!")
                    st.info(f"Processed {len(text)} characters in {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Book processing error")

# Main Content Tabs
if st.session_state.book_uploaded:
    learn_tab, quiz_tab = st.tabs(["Learning", "Quiz"])
    
    with learn_tab:
        st.header("Ask Questions")
        question = st.text_input("Ask about the book content:", key="question_input")
        
        if question and st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    # Verify QA system is initialized
                    if not hasattr(st.session_state.rag_service, 'qa_chain') or st.session_state.rag_service.qa_chain is None:
                        st.error("QA system not initialized. Please process the book again.")
                        st.stop()
                    
                    answer = st.session_state.rag_service.query(question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("QA query error")
    
    with quiz_tab:
        st.header("Test Your Knowledge")
        
        if not st.session_state.quiz_generated:
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz questions. This may take a minute..."):
                    try:
                        quiz_data = st.session_state.quiz_service.generate_quiz(st.session_state.book_text)
                        st.session_state.quiz_data = quiz_data
                        st.session_state.quiz_generated = True
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_result = None
                        st.success("Quiz generated!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.exception("Quiz generation error")
        
        if st.session_state.quiz_generated and st.session_state.quiz_data:
            quiz = st.session_state.quiz_data
            
            with st.form(key="quiz_form"):
                st.subheader("Quiz Questions")
                for q in quiz["questions"]:
                    # Create a unique key for each question
                    question_key = f"q_{q['id']}"
                    
                    # Initialize answer in session state if not exists
                    if question_key not in st.session_state.user_answers:
                        st.session_state.user_answers[question_key] = ""
                    
                    st.markdown(f"**Q{q['id']}:** {q['question']}")
                    
                    # Display options as radio buttons
                    options = q['options']
                    selected = st.radio(
                        f"Select answer for Q{q['id']}:",
                        options=list(options.keys()),
                        format_func=lambda k: f"{k}. {options[k]}",
                        key=f"radio_{q['id']}",
                        index=None  # No default selection
                    )
                    
                    # Store selected answer
                    st.session_state.user_answers[question_key] = selected if selected else ""
                
                # Form submit button
                submitted = st.form_submit_button("Submit Quiz")
                if submitted:
                    st.session_state.quiz_submitted = True
        
        if st.session_state.get("quiz_submitted", False):
            with st.spinner("Evaluating answers..."):
                try:
                    # Prepare answers in {question_id: answer} format
                    answers_to_send = {}
                    for key, value in st.session_state.user_answers.items():
                        if key.startswith("q_"):
                            question_id = key.split("_")[1]
                            answers_to_send[question_id] = value
                    
                    result = st.session_state.quiz_service.evaluate_quiz(
                        st.session_state.quiz_data["quiz_id"],
                        answers_to_send
                    )
                    st.session_state.quiz_result = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Quiz evaluation error")
            
            if st.session_state.get("quiz_result"):
                result = st.session_state.quiz_result
                st.success(f"## Your Score: {result['score']}")
                
                with st.expander("Detailed Results"):
                    for res in result["detail"]:
                        status = "✅" if res["is_correct"] else "❌"
                        st.markdown(f"{status} **Question {res['question_id']}:** {res['question']}")
                        st.markdown(f"- Your answer: **{res['user_answer']}**")
                        st.markdown(f"- Correct answer: **{res['correct_answer']}**")
                        
                        # Show options for reference
                        options = res.get('options', {})
                        if options:
                            st.markdown("Options:")
                            for opt, text in options.items():
                                prefix = "✓ " if opt == res['correct_answer'] else "  "
                                st.markdown(f"{prefix}**{opt}**: {text}")
                        
                        st.divider()

# Initial message
if not st.session_state.book_uploaded:
    st.info("📘 Please upload a book PDF to get started")
    
    # Debug information
    with st.expander("Debug Info"):
        st.write("Session state keys:", list(st.session_state.keys()))
        if 'rag_service' in st.session_state:
            rag_status = "Initialized" if hasattr(st.session_state.rag_service, 'qa_chain') else "Not initialized"
            st.write(f"RAG Service: {rag_status}")
        else:
            st.write("RAG Service: Not created")