"""
消息简写形式：与05对比，使用元组(tuple)代替Message类
LangChain自动将元组转换为对应的Message类型，提升代码简洁性
"""

from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 简写形式：(角色, 内容)
# 角色可选值：system/human/ai（大小写敏感）
# LangChain内部自动调用对应Message类进行转换
messages = [
    ("system", "你是一个边塞诗人。"),
    ("human", "写一首唐诗。"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照你上一个回复的格式，在写一首唐诗。")
]

# LangChain自动将元组转换为对应Message类型
res = model.stream(input=messages)

for chunk in res:
    print(chunk.content, end="", flush=True)
