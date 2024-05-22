import os
from dotenv import load_dotenv

from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever

load_dotenv()

ES_URL = os.getenv('ES_URL')
ES_USER = os.getenv('ES_USER')
ES_PASSWORD = os.getenv('ES_PASSWORD')

index_name = "test-langchain-retriever"
text_field = "text"

def bm25_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {
                text_field: search_query,
            },
        },
    }


bm25_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=bm25_query,
    content_field=text_field,
    url=ES_URL,
    username=ES_USER,
    password=ES_PASSWORD
)

print('============================================================================')
print(bm25_retriever.invoke("foo"))
print('============================================================================')
