from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
  Answer the question below.
  Here is the conversation history: {context}
  Question: {question}
  Answer: 
"""

model = OllamaLLM(model="gemma3:1b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# result = model.invoke(input = "Hello world!")
# print(result)

def handle_conversation():
  context = ""
  print("Welcome to the conversation! Type 'exit' to end.")
  while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
      break
    result = chain.invoke(
      {
        "context": context,
        "question": user_input
      }
    )
    print("Gemma: ", result)
    context += f"User: {user_input}\nAI: {result}\n"

if __name__ == "__main__":
  handle_conversation()