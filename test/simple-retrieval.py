import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI model instance
llm = ChatOpenAI(api_key=my_openai_api_key)

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

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Define the chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Invoke the chain
print(document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
}))

# Define the retriever and retrieve chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...
