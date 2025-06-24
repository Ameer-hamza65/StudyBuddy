import streamlit as st
import requests
import json
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="StudyBuddy", layout="wide")

# Initialize session state
def init_session():
    if "book_uploaded" not in st.session_state:
        st.session_state.book_uploaded = False
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

init_session()

# UI Components
st.title("📚 StudyBuddy")
st.subheader("Learn from books and test your knowledge")

# Sidebar for book upload
# In the sidebar section:
with st.sidebar:
    st.header("Upload Book")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
    
    if uploaded_file is not None and not st.session_state.book_uploaded:
        # Validate file type on frontend
        if uploaded_file.type != "application/pdf" and not uploaded_file.name.lower().endswith(".pdf"):
            st.error("Please upload a valid PDF file")
            st.stop()
        
        if st.button("Process Book"):
            with st.spinner("Processing book..."):
                try:
                    # Send as multipart form data
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(
                        f"{BACKEND_URL}/upload-book/", 
                        files=files
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
                response = requests.post(
                    f"{BACKEND_URL}/ask/",
                    json={"question": question}
                )
                if response.status_code == 200:
                    st.markdown(f"**Answer:** {response.json()['answer']}")
                else:
                    st.error("Failed to get answer")
    
    with quiz_tab:
        st.header("Test Your Knowledge")
        
        if not st.session_state.quiz_generated:
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz questions..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/generate-quiz/")
                        if response.status_code == 200:
                            quiz_data = response.json()
                            st.session_state.quiz_data = quiz_data
                            st.session_state.quiz_generated = True
                            st.session_state.user_answers = {}
                            st.success("Quiz generated!")
                        else:
                            st.error(f"Failed to generate quiz: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        
        if st.session_state.quiz_generated:
            quiz = st.session_state.quiz_data
            
            with st.form("quiz_form"):
                st.subheader("Quiz Questions")
                for q in quiz["questions"]:
                    st.markdown(f"**Q{q['id']}:** {q['question']}")
                    
                    # Display options
                    options = q['options']
                    selected = st.radio(
                        f"Select answer for Q{q['id']}:",
                        options=list(options.keys()),
                        format_func=lambda k: f"{k}. {options[k]}",
                        key=f"q_{q['id']}",
                        index=None
                    )
                    
                    # Store answer
                    st.session_state.user_answers[str(q['id'])] = selected if selected else ""
                
                submit = st.form_submit_button("Submit Quiz")
                if submit:
                    st.session_state.quiz_submitted = True
            
            if st.session_state.get("quiz_submitted", False):
                with st.spinner("Evaluating answers..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/evaluate-quiz/",
                            json={
                                "quiz_id": quiz["quiz_id"],
                                "answers": st.session_state.user_answers
                            }
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.quiz_result = result
                        else:
                            st.error(f"Evaluation failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                
                if "quiz_result" in st.session_state:
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