from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "prospecting_agent"

    # API Keys
    SERPER_API_KEY: str
    APOLLO_API_KEY: str

    # Application
    APP_NAME: str = "Prospecting Agent"
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
