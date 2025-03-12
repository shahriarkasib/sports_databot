import concurrent.futures
import threading
from langchain_openai import ChatOpenAI

# Create a single ChatOpenAI instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

def worker(query):
    try:
        response = llm.invoke(query)
        print(f"Thread {threading.get_ident()} got response: {response}")
        return response
    except Exception as e:
        print(f"Thread {threading.get_ident()} encountered error: {e}")
        return None

# Define several test queries
queries = [
    "What is the capital of France?",
    "Tell me a joke.",
    "How many planets are in the solar system?",
    "What is the weather like today?",
    "Who wrote 'Pride and Prejudice'?"
]

# Use ThreadPoolExecutor to run the worker function concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(worker, queries))

print("Test results:", results)
