from typing import TypedDict, List, Annotated
import operator

class GraphState(TypedDict):
    query: str
    answers: Annotated[List[str], operator.add]