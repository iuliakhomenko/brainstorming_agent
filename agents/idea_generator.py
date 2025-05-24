from langchain_core.prompts import ChatPromptTemplate
from config import llm


# --- Idea Generator --- #
class IdeaGenerator:
    def __call__(self, state):
        topic = state["topic"]
        technique = state["technique"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Use {technique} to generate 5 creative ideas on the topic."),
            ("human", f"Topic: {topic}")
        ])
        ideas = llm.invoke(prompt.format_messages())
        return {"ideas": ideas.content, **state}
