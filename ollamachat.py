import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

##Prompt Template
prompt=ChatPromptTemplate.from_messages(
[
    ("system","You behave as a Sr. AI consultant and answer following"),
    ("user","Question:{question}")
]
)

#streamlit
st.title("Langchin demo with llama2")
input_text=st.text_input("What Question you have in mind?")

#OLLAMA
llm=ChatOllama(model="gemma:2b")
output_parser=StrOutputParser()

chain=prompt | llm |output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))


