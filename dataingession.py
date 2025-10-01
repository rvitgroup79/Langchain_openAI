##Data Ingession : Lanchain document loader

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import os
from dotenv import load_dotenv
from langchainopenai import OpenAIEmbeddings



##loader=TextLoader('rvdata.txt')  
##content=loader.load()
##print(content)


#pdf Document Loader
'''
docs=PyPDFLoader('ProposalFitness.pdf')
content=docs.load()
print('PDF File content')
print(content[1])
'''

#Website content Loder
'''
webloader=WebBaseLoader(web_path='https://rvtechnologies.com/ai-development-company.html',bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("custom-health_sec","AiBannerSection","readyfor-betterAI"))))
content=webloader.load()
print(content)
'''

#arxiv loader
'''
content=ArxivLoader(query="2509.09677", load_max_doc=2).load()
print(content)
'''

'''
#wikipedia
content=WikipediaLoader(query="Maharaja Ranjit Singh", load_max_docs=2).load()
#print(content)
'''
'''
## Text splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
final_document=text_splitter.split_documents(content)
print(final_document[0])
print("\n\n\n")
print(final_document[1])

'''

loader=TextLoader('rvdata.txt')
content=loader.load()
##print(content)

with open('rvdata.txt') as file:
    rvdata=file.read()
    ##print (rvdata)


text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
text=text_splitter.create_documents([rvdata])
##print(text[1])


##Embedding Techniques -openAI
load_dotenv()
os.environ["OPENAI_API_KEY"]= os.getenv('OPENAI_API_KEY')

embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)
text4embeding="RV Technologies Softwares pvt ltd, chandigarh, works in mobile app development,Website Design and development and digital marketing"
embedding_result=embedding.embed_query()
print (embedding_result)



