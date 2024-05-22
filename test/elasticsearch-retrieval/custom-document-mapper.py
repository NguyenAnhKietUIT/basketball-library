import os
from dotenv import load_dotenv

from typing import Any, Dict

from langchain_core.documents import Document
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

def num_characters_mapper(hit: Dict[str, Any]) -> Document:
    num_chars = hit["_source"][num_characters_field]
    content = hit["_source"][text_field]
    return Document(
        page_content=f"This document has {num_chars} characters",
        metadata={"text_content": content},
    )


custom_mapped_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=filter_query_func,
    document_mapper=num_characters_mapper,
    url=ES_URL,
    username=ES_USER,
    password=ES_PASSWORD
)

print('============================================================================')
print(custom_mapped_retriever.invoke("foo"))
print('============================================================================')
