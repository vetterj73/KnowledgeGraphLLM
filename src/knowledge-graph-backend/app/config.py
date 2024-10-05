# app/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    AZURE_CONNECTION_STRING: str = os.environ["AZURE_CONNECTION_STRING"]
    AZURE_CONTAINER_NAME: str = os.environ["AZURE_CONTAINER_NAME"]
    OPEN_API_KEY: str = os.environ["OPEN_API_KEY"]
    INDEX_PATH: str = os.environ["INDEX_PATH"]
    MAPPING_PATH: str = os.environ["MAPPING_PATH"]
    GRAPH_PATH: str = os.environ["GRAPH_PATH"]
    FILE_ENCODING: str = os.environ["FILE_ENCODING"]
    BATCH_SIZE_LIMIT: int = int(os.environ["BATCH_SIZE_LIMIT"])

    class Config:
        case_sensitive = True


settings = Settings()
