from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document


import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

llm=ChatOpenAI(model="gpt-4o")
loader=WebBaseLoader("https://rvtechnologies.com/ai-development-company.html")
rvcontent=loader.load()
#output_parser=StrOutputParser()

#print(type(rvcontent))

doc_splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
splitted_docs=doc_splitter.split_documents(rvcontent)
#print(splitted_docs)

#embedding
embedding=OpenAIEmbeddings()
#embed_result=embedding.embed_documents(splitted_docs)
vectorstoredb=FAISS.from_documents(splitted_docs,embedding)
#print(vectorstoredb)

'''#
query="Do RV Technologies provide services in Generative AI"
search_result=vectorstoredb.similarity_search(query)
#print(search_result[0].page_content)
'''

#retrieval chain and document chain
prompt=ChatPromptTemplate.from_template(
    """
    Answer the following based on given context
    <context>
    {context}
    </context>
"""
)

document_chain=create_stuff_documents_chain(llm,prompt)

retriever=vectorstoredb.as_retriever()

retrieval_chain =create_retrieval_chain(retriever,document_chain)



document_chain.invoke({"input":"Do RV Technologies provide services in Generative AI",
                       "context":[Document(page_content="RV Technologies build custom generative AI solutions leveraging state-of-the-art multimodal models like GPT-4o, LLaMA 3, Gemini, and Stable Diffusion. Our development stack includes LangChain, Hugging Face Transformers, and vector databases such as Pinecone and Weaviate.")]
                       })
response=retrieval_chain.invoke({"input":"Do RV Technologies provide services in Generative AI"})
print(response["answer"])