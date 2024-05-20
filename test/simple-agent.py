import os
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

# Define the loader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

# Define the documents
docs = loader.load()

# Define the embedding
embeddings = OpenAIEmbeddings()

# Indexing the documents
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Define the retriever
retriever = vector.as_retriever()

# Define the retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Define the search tool
search = TavilySearchResults()

# Define list of tools
tools = [retriever_tool, search]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Initialize the OpenAI model instance
llm = ChatOpenAI(api_key=my_openai_api_key, model="gpt-3.5-turbo", temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print('============================================================================')
print(agent_executor.invoke({"input": "how can langsmith help with testing?"}))
print('============================================================================')

print('============================================================================')
print(agent_executor.invoke({"input": "what is the weather in SF?"}))
print('============================================================================')

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

print('============================================================================')
agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print('============================================================================')