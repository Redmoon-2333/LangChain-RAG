"""
文件功能：LangChain流式输出（与02/03对比：使用stream方法）
"""

from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

res = model.stream(input="你是谁呀能做什么？")  # stream: 流式输出，返回生成器

for chunk in res:  # chunk: 每次迭代返回的部分响应
    print(chunk, end="", flush=True)  # flush=True: 立即刷新输出缓冲区
