from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    embed_model: str = "models/embedding-001"
    llm_model: str = "gemini-2.0-flash"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"

settings = Settings()