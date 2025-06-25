import streamlit as st
import requests
import json
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="StudyBuddy", layout="wide")

# Initialize session state
def init_session():
    session_defaults = {
        "book_uploaded": False,
        "quiz_generated": False,
        "quiz_submitted": False,
        "user_answers": {},
        "quiz_data": None,
        "quiz_result": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# UI Components
st.title("📚 StudyBuddy")
st.subheader("Learn from books and test your knowledge")

# Sidebar for book upload
with st.sidebar:
    st.header("Upload Book")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
    
    if uploaded_file is not None and not st.session_state.book_uploaded:
        if st.button("Process Book"):
            with st.spinner("Processing book..."):
                try:
                    # Send as multipart form data
                    response = requests.post(
                        f"{BACKEND_URL}/upload-book/", 
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300  # 5-minute timeout
                    )
                    if response.status_code == 200:
                        st.session_state.book_uploaded = True
                        st.success("Book processed successfully!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

# Main Content Tabs
if st.session_state.book_uploaded:
    learn_tab, quiz_tab = st.tabs(["Learning", "Quiz"])
    
    with learn_tab:
        st.header("Ask Questions")
        question = st.text_input("Ask about the book content:", key="question_input")
        
        if question and st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/ask/",
                        json={"question": question},
                        timeout=60
                    )
                    if response.status_code == 200:
                        st.markdown(f"**Answer:** {response.json()['answer']}")
                    else:
                        st.error("Failed to get answer")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    with quiz_tab:
        st.header("Test Your Knowledge")
        
        if not st.session_state.quiz_generated:
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz questions..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/generate-quiz/",
                            timeout=120
                        )
                        if response.status_code == 200:
                            st.session_state.quiz_data = response.json()
                            st.session_state.quiz_generated = True
                            st.session_state.user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.success("Quiz generated!")
                        else:
                            st.error(f"Failed to generate quiz: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        
        if st.session_state.quiz_generated and st.session_state.quiz_data:
            quiz = st.session_state.quiz_data
            
            # Create form for quiz questions
            with st.form(key="quiz_form"):
                st.subheader("Quiz Questions")
                
                for q in quiz["questions"]:
                    # Create a unique key for each question
                    question_key = f"q_{q['id']}"
                    
                    # Initialize answer in session state if not exists
                    if question_key not in st.session_state.user_answers:
                        st.session_state.user_answers[question_key] = None
                    
                    st.markdown(f"**Q{q['id']}:** {q['question']}")
                    
                    # Display options as radio buttons
                    options = q['options']
                    selected = st.radio(
                        f"Select answer for Q{q['id']}:",
                        options=list(options.keys()),
                        format_func=lambda k: f"{k}. {options[k]}",
                        key=f"radio_{q['id']}",
                        index=0  # Default to first option
                    )
                    
                    # Store selected answer
                    st.session_state.user_answers[question_key] = selected
                
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
                    
                    response = requests.post(
                        f"{BACKEND_URL}/evaluate-quiz/",
                        json={
                            "quiz_id": st.session_state.quiz_data["quiz_id"],
                            "answers": answers_to_send
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        st.session_state.quiz_result = response.json()
                    else:
                        st.error(f"Evaluation failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
            
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

# Initial state message
if not st.session_state.book_uploaded:
    st.info("📘 Please upload a book PDF to get started")