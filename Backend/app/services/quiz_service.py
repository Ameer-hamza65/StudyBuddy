import json, time
import re
import random
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from app.utils.config import settings
from pydantic import BaseModel, ValidationError

# Configure logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class Question(BaseModel):
    id: int
    question: str
    options: dict  # {"A": "option text", "B": ...}
    correct: str

class QuizData(BaseModel):
    quiz_id: str
    questions: list[Question]

class QuizService:
    def __init__(self):
        self.quizzes = {} 
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.7 
        )
        self.setup_chains()

    def setup_chains(self):
        """Setup quiz generation and evaluation chains"""
        # Quiz generation template
        quiz_template = """
        You are an expert MCQ maker. Given the following book content, create a quiz of exactly {number} multiple choice questions.
        Guidelines:
        1. Each question must have 4 options (A, B, C, D) with one correct answer
        2. Questions should cover key concepts from the text
        3. Avoid repetition and ensure questions are distinct
        4. Format response as valid JSON matching this structure:
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
        Book Content: {text}
        """
        
        self.quiz_prompt = PromptTemplate(
            input_variables=['text', 'number'],
            template=quiz_template
        )
        
        self.quiz_chain = LLMChain(
            llm=self.llm,
            prompt=self.quiz_prompt,
            output_key='quiz_json',
            verbose=True
        )
        
        # Quiz evaluation template
        evaluation_template = """
        You are an expert quiz evaluator. Given the following quiz questions and original book content:
        Book Content: {text}
        Quiz Questions: {quiz_json}
        
        Please perform these tasks:
        1. Ensure all questions are answerable from the book content
        2. Verify each question has exactly one correct answer
        3. Check that options are distinct and non-overlapping
        4. Return the validated quiz in the same JSON format
        """
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=['text', 'quiz_json'],
            template=evaluation_template
        )
        
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=self.evaluation_prompt,
            output_key='validated_quiz',
            verbose=True
        )
        
        # Combined chain - FIXED: Properly connect input/output variables
        self.quiz_generation_chain = SequentialChain(
            chains=[self.quiz_chain, self.evaluation_chain],
            input_variables=['text', 'number'],
            output_variables=['validated_quiz'],
            verbose=True
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

    def validate_quiz(self, quiz_data: dict) -> QuizData:
        """Validate quiz structure using Pydantic"""
        try:
            # Generate unique quiz ID
            quiz_id = f"quiz_{random.randint(1000,9999)}_{int(time.time())}"
            quiz_data["quiz_id"] = quiz_id
            
            # Validate with Pydantic
            quiz = QuizData(**quiz_data)
            return quiz
        except ValidationError as e:
            logger.error(f"Quiz validation failed: {e}")
            return None

    def generate_quiz(self, context: str, retries=3) -> dict:
        """Generate quiz questions from book context with validation"""
        for attempt in range(retries):
            try:
                result = self.quiz_generation_chain.invoke({
                    "text": context[:10000],  # Truncate for token limits
                    "number": 10
                })
                
                # Extract JSON from response
                quiz_json = self.extract_json(result['validated_quiz'])
                if not quiz_json:
                    raise ValueError("JSON extraction failed")
                
                # Validate structure
                quiz = self.validate_quiz(quiz_json)
                if quiz:
                    self.quizzes[quiz.quiz_id] = quiz
                    return quiz.dict()
            
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(1)  # Add delay between retries
        
        raise RuntimeError("Failed to generate valid quiz after multiple attempts")

    def evaluate_quiz(self, quiz_id: str, answers: dict) -> dict:
        """Evaluate user's quiz answers"""
        if quiz_id not in self.quizzes:
            raise KeyError("Quiz not found")
        
        quiz = self.quizzes[quiz_id]
        score = 0
        results = []
        
        for q in quiz.questions:
            # Get user answer (convert to uppercase)
            user_answer = answers.get(str(q.id), "").strip().upper()
            correct_answer = q.correct.strip().upper()
            is_correct = user_answer == correct_answer
            
            if is_correct:
                score += 1
                
            results.append({
                "question_id": q.id,
                "question": q.question,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "options": q.options
            })
        
        return {
            "score": f"{score}/{len(quiz.questions)}",
            "detail": results
        }