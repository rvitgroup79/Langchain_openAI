from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

llm=ChatOpenAI(model="gpt-4o")
#result=llm.invoke("what is the future of software developers after AI tools?, what sof")
#print(result)
prompt= ChatPromptTemplate.from_messages(
    [
        ("system","You are a IT Technical expert who have the knowledge of software market, answer users question"),
        ("user","{input}")
    ]
)

output_parser=StrOutputParser()
chain=prompt|llm|output_parser
response=chain.invoke({"input":"what is the future of software developers after AI tools, what software developer should do"})
print(response)