"""
Configuration Settings.

This module defines the application configuration using Pydantic's BaseSettings.
It handles environment variable loading and provides typed access to configuration values.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings model.

    Attributes:
        API_V1_STR (str): The prefix for V1 API endpoints.
        PROJECT_NAME (str): The name of the project/application.
        DATABASE_URL (str): The connection string for the database.
    """
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "GearMeshing-AI Server"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/gearmeshing_ai"
    
    # Security / Auth (Placeholders for now, not strictly detailed in spec but good practice)
    # SECRET_KEY: str = "changeme"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

