from state import GraphState
from llm import load_chat_llm

llm = load_chat_llm()

def node_a(state: GraphState) -> GraphState:
    ans = llm.invoke(f"Upper-case this: {state['query']}").content.strip()
    return {"answers": [f"A: {ans}"]}

def node_b(state: GraphState) -> GraphState:
    ans = llm.invoke(f"Lower-case this: {state['query']}").content.strip()
    return {"answers": [f"B: {ans}"]}

def node_c(state: GraphState) -> GraphState:
    ans = llm.invoke(f"Translate to Spanish: {state['query']}").content.strip()
    return {"answers": [f"C: {ans}"]}