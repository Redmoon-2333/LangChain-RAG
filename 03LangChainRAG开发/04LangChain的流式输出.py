"""
LangChain流式输出：与同步调用(invoke)对比，使用stream方法实现实时流式响应
原理：模型生成token后立即返回，无需等待完整响应，提升用户体验
"""

from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

# stream: 流式输出方法，返回生成器(Generator)而非完整字符串
# 每次迭代返回模型新生成的片段(chunk)，实现逐字打印效果
res = model.stream(input="你是谁呀能做什么？")

for chunk in res:  # chunk: 每次迭代返回的部分响应
    print(chunk, end="", flush=True)  # flush=True: 立即刷新输出缓冲区，确保实时显示
