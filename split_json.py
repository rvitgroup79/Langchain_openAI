import json
import requests
from langchain_text_splitters import RecursiveJsonSplitter
json_data=requests.get('https://api.smith.langchain.com/openapi.json').json()
json_splitter=RecursiveJsonSplitter(max_chunk_size=300)
json_chunks=json_splitter.split_json(json_data)

for chunk in json_chunks[:3]:
    print(chunk)

