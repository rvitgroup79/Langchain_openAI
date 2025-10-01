from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter


##Text File Loader
loader=TextLoader('rvdata.txt')
content=loader.load()

##Text splitter
text_splitter=CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs=text_splitter.split_documents(content)
#print(docs)


#embedding :Ollama
embedding = OllamaEmbeddings(model="gemma:2b")

##Store document in FAISS vector DB
db = FAISS.from_documents(docs, embedding)

#print(db)
query="Where RV Technologies is located?"
final_docs=db.similarity_search(query)
#print (final_docs[0].page_content)

#Retriever : Convert Vector database into Retriever class to use it with any LLM model
#retriever=db.as_retriever()
#retriever_docs=retriever.invoke(query)
#print(docs[0].page_content)


docs_scores=db.similarity_search_with_score(query)
print(docs_scores)
