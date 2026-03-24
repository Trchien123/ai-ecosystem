from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = ROOT_DIR / ".env"

class Settings(BaseSettings):
    """
    Using Pydantic to validate the .env data and create the configuration for
    the whole system.
    """
    # API Keys
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PUSHOVER_USER: str
    PUSHOVER_TOKEN: str
    PUSHOVER_URL: str

    # AI Model Settings
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSIONS: int

    # Database Setting
    PINECONE_INDEX_NAME: str
    MONGO_DB_URI: str
    MONGO_DB_NAME: str

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings():
    """ Create Singleton instance for the Settings"""
    return Settings()

# Instance for the whole project
settings = get_settings()
