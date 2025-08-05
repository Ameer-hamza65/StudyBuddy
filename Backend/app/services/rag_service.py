from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.utils.config import settings

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None  # Don't initialize here

    async def initialize_qa_system(self, chunks: list[str]):
        """Initialize vector store and QA system"""
        # Initialize embeddings here, inside async context
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embed_model
        )

        self.vector_store = Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        self.vector_store.persist()

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

    def query(self, question: str) -> str:
        """Answer question based on book context"""
        if not self.qa_chain:
            raise ValueError("QA system not initialized")
        result = self.qa_chain.invoke({"query": question})
        return result["result"]
