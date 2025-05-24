# Coordinates technique, LLM, evaluator
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from technique_selector import TechniqueSelector
from idea_evaluator import IdeaEvaluator
from idea_generator import IdeaGenerator
from core.output_formatter import OutputFormatter



# --- Graph Definition --- #
def build_brainstorming_graph():
    selector = TechniqueSelector()
    generator = IdeaGenerator()
    evaluator = IdeaEvaluator()
    formatter = OutputFormatter()

    graph = StateGraph()
    graph.add_node("select_technique", RunnableLambda(selector))
    graph.add_node("generate_ideas", RunnableLambda(generator))
    graph.add_node("evaluate_ideas", RunnableLambda(evaluator))
    graph.add_node("format_output", RunnableLambda(formatter))

    graph.set_entry_point("select_technique")
    graph.add_edge("select_technique", "generate_ideas")
    graph.add_edge("generate_ideas", "evaluate_ideas")
    graph.add_edge("evaluate_ideas", "format_output")
    graph.set_finish_point("format_output")

    return graph.compile()


# --- Example Invocation --- #
if __name__ == "__main__":
    graph = build_brainstorming_graph()
    result = graph.invoke({"topic": "Sustainable packaging for e-commerce"})
    print(result["final_output"])
