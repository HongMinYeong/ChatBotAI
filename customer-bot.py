import dotenv
dotenv.load_dotenv()
from langchain.document_loaders import WebBaseLoader #웹사이트 내에 html 정보 불러올수있게 하는 함수 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# import streamlit as st
# import time

loader = WebBaseLoader("https://dalpha.so/ko/howtouse?scrollTo=custom") #웹사이트 내의 텍스트들
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    retriever, 
    "customer_service", #tool 의 제목 
    "Searches and return documents regarding the customer service guide.", #설명 
)
tools = [tool]

