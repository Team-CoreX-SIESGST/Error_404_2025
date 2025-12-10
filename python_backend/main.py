# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph   # your existing module

# ------------------ schema ------------------ #
class QueryIn(BaseModel):
    query: str

class AnswerOut(BaseModel):
    answers: list[str]          # adjust type if answers is not list[str]

# ------------------ app init ------------------ #
app = FastAPI(title="LangGraph service")

chain = build_graph()           # build once, reuse for every request

# ------------------ endpoint ------------------ #
@app.post("/ask", response_model=AnswerOut)
def ask(payload: QueryIn):
    """
    Run the LangGraph chain and return the final answers.
    """
    result = chain.invoke({"query": payload.query})
    return AnswerOut(answers=result["answers"])