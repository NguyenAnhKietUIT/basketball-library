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
num_characters_field = "num_characters"

def filter_query_func(search_query: str) -> Dict:
    return {
        "query": {
            "bool": {
                "must": [
                    {"range": {num_characters_field: {"gte": 5}}},
                ],
                "must_not": [
                    {"prefix": {text_field: "bla"}},
                ],
                "should": [
                    {"match": {text_field: search_query}},
                ],
            }
        }
    }


filtering_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=filter_query_func,
    content_field=text_field,
    url=ES_URL,
    username=ES_USER,
    password=ES_PASSWORD
)

print('============================================================================')
print(filtering_retriever.invoke("foo"))
print('============================================================================')
