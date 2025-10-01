from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader =TextLoader("rvdata.txt")
textdocs=loader.load()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
final_document=text_splitter.split_documents(textdocs)
#print (final_document)

#embedding1 = OllamaEmbeddings(model="gemma:2b")

embedding1=(
    OllamaEmbeddings(model="gemma:2b")
)

vectordb=Chroma.from_documents(documents=final_document, embedding=embedding1)

#print(vectordb)

query="do Rv Technologies also provide training"
docs=vectordb.similarity_search(query)
print(docs[0].page_content)


