from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # API Keys
    NCBI_API_KEY: str = Field(default="", env="NCBI_API_KEY")
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    LANGSMITH_API_KEY: str = Field(default="", env="LANGSMITH_API_KEY")
    OPENROUTER_API_KEY: str = Field(default="", env="OPENROUTER_API_KEY")

    # Embedding
    EMBEDDING_MODEL: str = Field(
        default="NeuML/pubmedbert-base-embeddings",
        env="EMBEDDING_MODEL"
    )
    EMBEDDING_DIMENSION: int = Field(default=768, env="EMBEDDING_DIMENSION")

    # PubMed
    PUBMED_MAX_RESULTS: int = Field(default=100, env="PUBMED_MAX_RESULTS")
    PUBMED_BATCH_SIZE: int = Field(default=20, env="PUBMED_BATCH_SIZE")

    # Chunking
    CHUNK_SIZE: int = Field(default=512, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=50, env="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
