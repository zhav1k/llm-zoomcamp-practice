import requests 
import elasticsearch
from tqdm import tqdm

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(documents[0])

elastic_index_config = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

es = elasticsearch.Elasticsearch("http://localhost:9200")
es.options(ignore_status=[400,404]).indices.delete(index='courses')
es.indices.create(index='courses', body=elastic_index_config, ignore=400)

for doc in tqdm(documents):
    es.index(index="courses", body=doc)

query = "How do I execute a command on a Kubernetes pod?"

query_settings = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            # "filter": {
            #     "term": {
            #         "course": "data-engineering-zoomcamp"
            #     }
            # },
        }
    }
}

response = es.search(index="courses", body=query_settings)

for hit in response['hits']['hits']:
    print(f"Score: {hit['_score']}\nCourse: {hit['_source']['course']}\nQuestion: {hit['_source']['question']}\nText: {hit['_source']['text']}\n")


query = "How do I copy a file to a Docker container?"

query_settings = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            },
        }
    }
}

response = es.search(index="courses", body=query_settings)

for hit in response['hits']['hits']:
    print(f"Score: {hit['_score']}\nCourse: {hit['_source']['course']}\nQuestion: {hit['_source']['question']}\nText: {hit['_source']['text']}\n")


context_template = """
Q: {question}
A: {text}
""".strip()

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

context_pieces = []

for hit in response['hits']['hits']:
    doc = hit['_source']
    context_piece = context_template.format(**doc)
    context_pieces.append(context_piece)

context = '\n\n'.join(context_pieces)

final_prompt = prompt_template.format(question=query, context=context)
print(final_prompt)

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
encoded_prompt = encoding.encode(final_prompt)
print(len(encoded_prompt))
print(encoding.encode(final_prompt))
