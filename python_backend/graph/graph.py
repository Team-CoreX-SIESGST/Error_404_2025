from langgraph.graph import StateGraph,START, END
from state import GraphState
from nodes import node_a, node_b, node_c

def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("A", node_a)
    builder.add_node("B", node_b)
    builder.add_node("C", node_c)

    # parallel fan-out from start
    builder.add_edge(START, "A")
    builder.add_edge(START, "B")
    builder.add_edge(START, "C")

    # converge to END
    builder.add_edge("A", END)
    builder.add_edge("B", END)
    builder.add_edge("C", END)

    return builder.compile()