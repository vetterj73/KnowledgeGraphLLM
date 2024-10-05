# app/models.py
from pydantic import BaseModel


class UserPrompt(BaseModel):
    prompt: str


class UploadResponse(BaseModel):
    message: str


class ProcessResponse(BaseModel):
    response: str


class ErrorResponse(BaseModel):
    detail: str
