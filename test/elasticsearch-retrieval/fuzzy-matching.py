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

def fuzzy_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {
                text_field: {
                    "query": search_query,
                    "fuzziness": "AUTO",
                }
            },
        },
    }


fuzzy_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=fuzzy_query,
    content_field=text_field,
    url=ES_URL,
    username=ES_USER,
    password=ES_PASSWORD
)
print('============================================================================')
print(fuzzy_retriever.invoke("fox"))  # note the character tolernace
print('============================================================================')
