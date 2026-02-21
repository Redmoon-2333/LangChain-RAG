"""
Ollama聊天模型 + 流式输出 + 消息历史
与05对比：使用Ollama本地模型作为后端，无需云端API
"""

# OllamaLLM同时支持LLM接口和Chat接口，内部自动适配
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = OllamaLLM(model="qwen3:4b")

# messages: 消息列表结构与ChatTongyi一致
messages = [
    SystemMessage(content="你是一个边塞诗人。"),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，在写一首唐诗。")
]

# 流式输出
# Warning: OllamaLLM的stream返回原始字符串，而ChatTongyi返回AIMessage类型
res = model.stream(input=messages)

for chunk in res:  # chunk是字符串（OllamaLLM的stream返回原始字符串）
    print(chunk, end="", flush=True)
