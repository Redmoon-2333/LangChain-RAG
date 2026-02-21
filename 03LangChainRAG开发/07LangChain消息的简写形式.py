"""
文件功能：消息简写形式（与05对比：用元组代替Message类）
"""

from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 简写：(角色, 内容)，角色: system/human/ai
messages = [
    ("system", "你是一个边塞诗人。"),
    ("human", "写一首唐诗。"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照你上一个回复的格式，在写一首唐诗。")
]

res = model.stream(input=messages)  # LangChain自动将元组转换为对应Message类型

for chunk in res:
    print(chunk.content, end="", flush=True)
