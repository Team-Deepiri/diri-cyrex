
from langchain_core.prompts import PromptTemplate


# Simple prompt template for our learning chain
LEARNING_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""You are a helpful learning assistant. The user asks: {user_input}

Think about the question and provide a helpful response. Keep it concise and educational.

Response:"""
)

