"""
StrOutputParser解析器：将模型输出从AIMessage转换为纯字符串
作用：提取AIMessage.content，简化后续处理，适用于纯文本输出场景
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化解析器：将AIMessage转为字符串
parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max-2026-01-23")
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字无需其它内容。"
)

# LCEL链式调用：支持多模型串联
# prompt → model → parser → model → parser
# 前一个组件的输出自动作为下一个组件的输入
chain = prompt | model | parser | model | parser

# invoke返回最终字符串类型
res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
print(type(res))
