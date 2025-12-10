import os
from langchain_google_genai import ChatGoogleGenerativeAI

# --- embed the key-handling here ---------------------------------
GOOGLE_API_KEY ="AIzaSyBjF70nHFMfJ0u1qiuODKQGsnbv7-t6EDw"
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not found.  \n"
        "Run: export GOOGLE_API_KEY='your-key'   (or set it in .env)"
    )

def load_chat_llm() -> ChatGoogleGenerativeAI:
    """Return Gemini-powered chat instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY   # explicit key
    )