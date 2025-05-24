# Configs: paths, LLM keys, DB, etc.
from langchain_openai import ChatOpenAI


# Global LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
