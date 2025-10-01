from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings


loader=TextLoader('rvdata.txt')
content=loader.load()
#print(content)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
final_text=text_splitter.split_documents(content)
#print(final_text)

load_dotenv()
'''
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
embedding_1024=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1024)

#Vector embeding in Vector Store chromaDB
db=Chroma.from_documents(final_text,embedding_1024)
#print(db)

query1="RV Technologies also runs a training institute"
retrieved_results=db.similarity_search(query1)
print(retrieved_results)
'''
embedding=(
    OllamaEmbeddings(model="gemma:2b")
)

r1=embedding.embed_documents(
    [
        'Alpha is the first letter of the Greek Alphabets',
        'Bita is the second letter of the Greek Alphabets',
    ]
)

embedding_result=embedding.embed_query("what is Bita")

print(embedding_result)
print(len(embedding_result))