import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']='true'

prompt=ChatPromptTemplate.from_messages(
[
    ("system","As a Sr. AI consultant"),
    ("user","Question:{question}")
]
)

#streamlit
st.title="OpenAI LLM chat intergation using langchain"
input_text=st.text_input("Ask anything you wat to ask?")

llm=ChatOpenAI(model="gpt-5")

output_parser=StrOutputParser()

chain=prompt|llm|output_parser
if input_text:
    st.write(chain.invoke({"question": input_text}))
