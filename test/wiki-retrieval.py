import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import WikipediaRetriever

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

retriever = WikipediaRetriever()

model = ChatOpenAI(api_key=my_openai_api_key, model="gpt-3.5-turbo")  # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "Do you know Stephen Curry",
    "I want to know about Bunhiacopxki"
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    print(f"-> **Result**: {result} \n")
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")