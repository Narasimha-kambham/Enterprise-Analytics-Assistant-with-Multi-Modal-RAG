import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GEMINI_MODEL

# Load environment variables from .env file
load_dotenv()

def get_gemini_model():
    """
    Returns a Gemini model instance with the API key from .env.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Ensure it is set in .env.")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )