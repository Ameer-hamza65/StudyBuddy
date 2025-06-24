from fastapi import FastAPI, UploadFile, HTTPException
from app.schemas import BookUpload, QuestionRequest, QuizRequest, QuizResponse, QuizResult
from app.services.rag_service import RAGService
from app.services.quiz_service import QuizService
from app.utils.file_processing import process_pdf, chunk_text
from app.utils.config import settings
import time

app = FastAPI()
rag_service = RAGService()
quiz_service = QuizService()
book_context = ""

@app.post("/upload-book/")
async def upload_book(file: UploadFile):
    global book_context
    # Improved file validation
    if not (file.filename and file.filename.lower().endswith('.pdf')):
        raise HTTPException(400, "Only PDF files are supported")
    
    try:
        file_bytes = await file.read()
        
        # Validate PDF magic number
        if file_bytes[:4] != b'%PDF':
            raise HTTPException(400, "Invalid PDF file format")
        
        text = process_pdf(file_bytes)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        rag_service.initialize_qa_system(chunks)
        book_context = text
        return {"message": "Book processed successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error processing book: {str(e)}")

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        answer = rag_service.query(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(500, f"Error processing question: {str(e)}")

@app.post("/generate-quiz/")
async def generate_quiz():
    if not book_context:
        raise HTTPException(400, "No book uploaded")
    try:
        quiz = quiz_service.generate_quiz(book_context)
        return quiz
    except Exception as e:
        raise HTTPException(500, f"Error generating quiz: {str(e)}")

@app.post("/evaluate-quiz/")
async def evaluate_quiz(request: QuizRequest):
    try:
        result = quiz_service.evaluate_quiz(request.quiz_id, request.answers)
        return QuizResult(
            score=result["score"],
            detail=result["detail"]
        )
    except Exception as e:
        raise HTTPException(500, f"Error evaluating quiz: {str(e)}")