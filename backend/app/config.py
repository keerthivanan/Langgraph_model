from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "LangGraph IAM Agent"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # LLM
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-small"

    # Memory / Checkpointer
    # "memory" = in-process MemorySaver (dev), "sqlite" = file-backed, "postgres" = production
    checkpointer_type: str = Field(default="memory", env="CHECKPOINTER_TYPE")
    sqlite_db_path: str = Field(default="./data/checkpoints.db", env="SQLITE_DB_PATH")
    postgres_url: str = Field(default="", env="POSTGRES_URL")

    # Vector DB
    # "inmemory" = FAISS local (dev), "pinecone" = cloud, "pgvector" = postgres extension
    vector_store_type: str = Field(default="inmemory", env="VECTOR_STORE_TYPE")
    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="langgraph-rag", env="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")

    # Long-term memory store
    long_term_memory_db: str = Field(default="./data/long_term_memory.db", env="LT_MEMORY_DB")

    # IAM / Auth
    jwt_secret_key: str = Field(default="langgraph-iam-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    token_expire_minutes: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
