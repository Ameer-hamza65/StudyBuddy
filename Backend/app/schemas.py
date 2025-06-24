from pydantic import BaseModel

class BookUpload(BaseModel):
    file_bytes: bytes

class QuestionRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    quiz_id: str
    answers: dict  # {question_id: answer}

class QuizResponse(BaseModel):
    quiz_id: str
    questions: list

class QuizResult(BaseModel):
    score: str
    detail: list