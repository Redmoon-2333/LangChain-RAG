"""
文件功能：LangChain聊天模型（ChatTongyi + 流式输出 + 消息历史）
"""

from langchain_community.chat_models.tongyi import ChatTongyi  # 聊天模型类，返回AIMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatTongyi(model="qwen3-max-2026-01-23")

messages = [
    SystemMessage(content="你是一个边塞诗人。"),  # 系统消息：设定模型角色
    HumanMessage(content="写一首唐诗"),          # 用户消息
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),  # AI历史回复
    HumanMessage(content="按照你上一个回复的格式，在写一首唐诗。")  # 利用历史上下文
]

res = model.stream(input=messages)  # 流式输出消息列表

for chunk in res:  # chunk是AIMessage类型
    print(chunk.content, end="", flush=True)


