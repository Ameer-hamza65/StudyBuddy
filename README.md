****📘 Project Overview: StudyBuddy**** <br>
**StudyBuddy** is an AI-powered learning companion built with FastAPI and Streamlit. It integrates two powerful modes:

You can try the StudyBuddy here: <br>
just upload the pdf from the top left corner <br>
https://studybuddyforyou.streamlit.app/

**How to Run**
1. cd Backend
2. pip install -r requirements.txt  
3. cd frontend
4. pip install -r requirements.txt
5. cd Backend <br>
6. uvicorn app.main:app --reload
7. Open a new terminal and than <br>
8. cd Frontend <br>
9. streamlit run app.py

1. Learning Route <br>
Retrieval-Augmented Generation (RAG) pipeline using LangChain and Gemini embeddings.

Users upload a book or document, and the AI answers questions based on the exact context, ensuring accurate, source-based responses.

2. Quiz Route <br>
Dynamically generates a 10-question multiple-choice quiz based on the uploaded book’s content.

Users answer the quiz via Streamlit, submit, and instantly receive a score (e.g., 7/10) with feedback highlighting correct and incorrect responses.

⚙️ Core Technologies <br>
FastAPI — lightweight, high-performance API to manage learning & quiz endpoints.

Streamlit — clean, intuitive UI for both reading/discussion and the quiz experience.

LangChain — orchestrates RAG workflows: embedding, retrieval, and generation.

Gemini Embeddings & Gemini Responses — powered by Google's advanced Gemini LLM for both understanding and question handling.

Testable & Modular — clean separation: one endpoint for “learn” interactions, one for “quiz” generation and scoring.

🎯 Key Features <br>
Upload any book or text file → ask it questions in the learning interface

Get a custom 10-question quiz derived directly from the source material

See your score and correct answers immediately after quiz submission

End-to-end solution: ingestion → context-aware Q&A → knowledge assessment

