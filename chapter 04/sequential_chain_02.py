import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.7)

prompt1 = PromptTemplate(
    input_variables=["review"],
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}",
)

prompt2 = PromptTemplate.from_template(
        "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)

prompt3 = PromptTemplate.from_template(
        "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. 숫자만 대답하세요.\n\n{translation}"
)

prompt4 = PromptTemplate.from_template(
        "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요.\n\n{review}"
)

prompt5 = PromptTemplate.from_template(
        "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요.\n답변 언어:{language}\n리뷰 요약:{summary}"
)

prompt6 = PromptTemplate.from_template(
        "다음 생성된 답변을 한국어로 번역해주세요. \n 리뷰 번역 {reply1}"
)

translate_chain_component = prompt1 | openai_llm | StrOutputParser()
summarize_chain_component = prompt2 | openai_llm | StrOutputParser()
sentiment_chain_component = prompt3 | openai_llm | StrOutputParser()
language_chain_component = prompt4 | openai_llm | StrOutputParser()
reply1_chain_component = prompt5 | openai_llm | StrOutputParser()
reply2_chain_component = prompt6 | openai_llm | StrOutputParser()

combined_lcel_chain = (
    RunnablePassthrough.assign(
        translation=lambda x: translate_chain_component.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        summary=lambda x: summarize_chain_component.invoke({"translation": x["translation"]}),
        sentiment_score=lambda x: sentiment_chain_component.invoke({"translation": x["translation"]}),
        language=lambda x: language_chain_component.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        reply1=lambda x: reply1_chain_component.invoke({"language": x["language"], "summary": x["summary"]})
    )
    | RunnablePassthrough.assign(
        reply2=lambda x: reply2_chain_component.invoke({"reply1": x["reply1"]})
    )
)

review_text = """
The hotel was clean and the staff were very helpful.
The location was convenient, close to many attractions.
However, the room was a bit small and the breakfast options were limited.
Overall, a decent stay but there is room for improvement.
"""

# 체인 실행 및 결과 출력
try:
   # .invoke() 메서드에 초기 입력을 딕셔너리 형태로 전달합니다.
   result = combined_lcel_chain.invoke(input={'review': review_text})

   # 결과 딕셔너리에서 각 키를 사용하여 값을 출력합니다.
   print(f'translation 결과: {result.get("translation", "N/A")} \n')
   print(f'summary 결과: {result.get("summary", "N/A")} \n')
   print(f'sentiment_score 결과: {result.get("sentiment_score", "N/A")} \n')
   print(f'language 결과: {result.get("language", "N/A")} \n')
   print(f'reply1 결과: {result.get("reply1", "N/A")} \n') 
   print(f'reply2 결과: {result.get("reply2", "N/A")} \n')
except Exception as e:
   print(f"Error: {e}")
