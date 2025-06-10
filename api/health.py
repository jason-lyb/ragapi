from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "normal"}

@router.get("/hello")
def hello():
    return {"message": "Hello, world!"}
