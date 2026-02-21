# langchain_ollama: LangChain集成Ollama的官方包
# Ollama: 本地大模型运行框架，无需联网即可使用本地部署的模型
from langchain_ollama import OllamaLLM

# 初始化本地模型：model指定本地部署的模型名称
# 前提：需提前通过ollama pull qwen3:4b 下载模型到本地
model = OllamaLLM(model="qwen3:4b")

# invoke参数可以是字符串或消息列表
# 与云端API对比：本地推理，延迟取决于本地硬件配置
res = model.invoke("你是谁呀能做什么？")

print(res)
