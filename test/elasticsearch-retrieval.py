import os
from dotenv import load_dotenv

from typing import Any, Dict, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever

es_url = os.getenv('ES_URL')
es_api_key = os.getenv('ES_API_KEY')

es_client = Elasticsearch(hosts=[es_url], api_key=es_api_key)
print(es_client.info())

# embeddings = DeterministicFakeEmbedding(size=3)