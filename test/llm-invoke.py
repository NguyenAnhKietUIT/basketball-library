import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI model instance
llm = ChatOpenAI(api_key=my_openai_api_key)

# Invoke the OpenAI model
print('============================================================================')
print(llm.invoke("how can langsmith help with testing?"))
print('============================================================================')
