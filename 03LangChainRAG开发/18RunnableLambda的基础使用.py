"""
RunnableLambda基础使用：将普通Python函数转换为LangChain的Runnable接口
作用：在Chain中实现自定义数据转换逻辑，扩展LangChain的灵活性
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max-2026-01-23")
str_parser = StrOutputParser()

# 第一个提示词：要求模型返回名字
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，仅生成一个名字，并告知我名字，不要额外信息。"
)

# 第二个提示词：解析名字含义
second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

# RunnableLambda: 将普通函数转换为Runnable接口
# 语法：将函数用圆括号包裹 (lambda x: ...) 或 (函数名)
# 原理：函数的输入是上一个组件的输出，输出传递给下一个组件
chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "曹", "gender": "女孩"}):
    print(chunk, end="", flush=True)
