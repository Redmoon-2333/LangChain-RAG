from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max-2026-01-23")
str_parser = StrOutputParser()

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，仅生成一个名字，并告知我名字，不要额外信息。"
)

second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

# RunnableLambda: 将普通函数转换为Runnable接口
# 函数的入参：AIMessage → dict ({"name": "xxx"})
# lambda ai_msg: {"name": ai_msg.content} 将AIMessage内容提取为字典
chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "曹", "gender": "女孩"}):
    print(chunk, end="", flush=True)
