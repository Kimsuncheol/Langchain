from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#지난 대화 히스토리 출력
if 'response_id' in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])

prompt = st.chat_input("물어보고 싶은 것을 입력하세요!")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if 'response_id' not in st.session_state:
        with st.spinner('Wait for it...'):
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions="당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
                input=prompt,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["{벡터 저장소 ID}"]
                }]
            )

    else:
        with st.spinner('Wait for it...'):
            response = client.responses.create(
                previous_response_id=st.session_state.response_id,
                model="gpt-4o-mini",
                instructions="당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
                input=prompt,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["{벡터 저장소 ID}"]
                }]
            )
                
    with st.chat_message('assistant'):
        st.write(response.output_text)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response.output_text})
    st.session_state.response_id = response.id