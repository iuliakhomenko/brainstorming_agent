# SCAMPER, 6 Hats, etc.
from langchain_core.prompts import ChatPromptTemplate
from config import llm

# --- Technique Selector --- #
class TechniqueSelector:
    def __call__(self, state):
        topic = state["topic"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in brainstorming methods."),
            ("human", f"Suggest the most suitable brainstorming technique for the topic: {topic}")
        ])
        technique = llm.invoke(prompt.format_messages())
        return {"technique": technique.content, **state}