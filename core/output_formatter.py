# Markdown/HTML/JSON formatting

# --- Output Formatter --- #
class OutputFormatter:
    def __call__(self, state):
        return {
            "final_output": f"Technique: {state['technique']}\n\nIdeas:\n{state['ideas']}\n\nEvaluation:\n{state['evaluation']}"
        }
