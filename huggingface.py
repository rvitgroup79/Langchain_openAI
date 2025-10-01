import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
load_dotenv()
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

text="this is a hugging face embadding tutorial"
query_result=embedding.embed_query(text)
print(query_result)
