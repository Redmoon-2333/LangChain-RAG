"""
LangChain聊天模型：与基础LLM模型对比，聊天模型(ChatModel)支持消息历史上下文
核心类：ChatTongyi（阿里云聊天模型），返回AIMessage对象而非纯字符串
"""

# 引入聊天模型类：ChatTongyi返回AIMessage类型
from langchain_community.chat_models.tongyi import ChatTongyi
# 引入消息类型：HumanMessage(用户)、AIMessage(AI)、SystemMessage(系统)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 初始化聊天模型
model = ChatTongyi(model="qwen3-max-2026-01-23")

# messages: 消息列表，支持多轮对话历史上下文
# SystemMessage: 系统消息，设定模型角色/行为约束
# HumanMessage: 用户消息
# AIMessage: AI历史回复，用于维护对话连贯性
messages = [
    SystemMessage(content="你是一个边塞诗人。"),  # 系统消息：设定模型角色
    HumanMessage(content="写一首唐诗"),          # 用户消息
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),  # AI历史回复
    HumanMessage(content="按照你上一个回复的格式，在写一首唐诗。")  # 利用历史上下文
]

# stream: 流式输出消息列表，chunk是AIMessage类型
res = model.stream(input=messages)

for chunk in res:  # chunk是AIMessage类型
    print(chunk.content, end="", flush=True)


