"""
文件功能：Ollama本地嵌入模型（与08对比：本地部署）
"""

from langchain_ollama import OllamaEmbeddings

model = OllamaEmbeddings(model="qwen3-embedding:4b")  # 指定本地embedding模型

print(model.embed_query("我喜欢你"))  # 单句向量化
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))  # 批量向量化
