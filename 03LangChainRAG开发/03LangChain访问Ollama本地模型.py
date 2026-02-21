# langchain_ollama: LangChain集成Ollama的官方包
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")  # model指定本地部署的模型名称

res = model.invoke("你是谁呀能做什么？")  # invoke参数可以是字符串或消息列表

print(res)
