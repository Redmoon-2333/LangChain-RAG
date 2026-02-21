"""
文件功能：Ollama聊天模型 + 流式输出 + 消息历史（与05对比：Ollama后端）
"""

from langchain_ollama import OllamaLLM  # OllamaLLM同时支持LLM和Chat接口
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = OllamaLLM(model="qwen3:4b")

messages = [
    SystemMessage(content="你是一个边塞诗人。"),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，在写一首唐诗。")
]

res = model.stream(input=messages)  # 流式输出

for chunk in res:  # chunk是字符串（OllamaLLM的stream返回原始字符串）
    print(chunk, end="", flush=True)
