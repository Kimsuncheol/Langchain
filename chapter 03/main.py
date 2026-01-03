from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])
    
output_parser = StrOutputParser()
    
chain = prompt | llm | output_parser
result = chain.invoke({"input": "Hello, how are you?"})
print(result)