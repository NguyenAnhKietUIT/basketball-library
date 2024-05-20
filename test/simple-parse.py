import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

# Initialize the OpenAI model instance
llm = ChatOpenAI(api_key=my_openai_api_key)

# Define the output parser
output_parser = StrOutputParser()

# Define the chain
chain = prompt | llm | output_parser

# Invoke the chain
print(chain.invoke({"input": "how can langsmith help with testing?"}))