# Heuristic, ML, or LLM-based evaluator
from langchain_core.prompts import ChatPromptTemplate
from config import llm


# --- Idea Evaluator --- #
class IdeaEvaluator:
    def __call__(self, state):
        ideas = state["ideas"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical evaluator of brainstorming ideas."),
            ("human", f"Evaluate these ideas based on originality and feasibility: {ideas}")
        ])
        evaluation = llm.invoke(prompt.format_messages())
        return {"evaluation": evaluation.content, **state}