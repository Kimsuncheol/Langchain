from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

st.title("인공지능 시인")

content = st.text_input("시의 주제를 제시해주세요")
st.write("시의 주제는", content)

if st.button("시 작성 요청하기"):
  with st.spinner('Wait for it...'):
      result = chain.invoke({"input": content + "에 대한 시를 써줘"})
      st.write(result)