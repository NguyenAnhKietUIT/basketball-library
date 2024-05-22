import os
from dotenv import load_dotenv

from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_community.embeddings import DeterministicFakeEmbedding

load_dotenv()

ES_URL = os.getenv('ES_URL')
ES_USER = os.getenv('ES_USER')
ES_PASSWORD = os.getenv('ES_PASSWORD')

embeddings = DeterministicFakeEmbedding(size=3)

index_name = "test-langchain-retriever"
text_field = "text"
dense_vector_field = "fake_embedding"

def hybrid_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)  # same embeddings as for indexing
    return {
        "query": {
            "match": {
                text_field: search_query,
            },
        },
        "knn": {
            "field": dense_vector_field,
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        },
        "rank": {"rrf": {}},
    }


hybrid_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=hybrid_query,
    content_field=text_field,
    url=ES_URL,
    username=ES_USER,
    password=ES_PASSWORD
)

print('============================================================================')
print(hybrid_retriever.invoke("foo"))
print('============================================================================')
