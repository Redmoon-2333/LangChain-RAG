"""
Ollama本地嵌入模型：与08对比，使用本地部署的嵌入模型
优势：无需联网，保护数据隐私，降低API调用成本
"""

from langchain_ollama import OllamaEmbeddings

# 初始化本地嵌入模型：model指定本地部署的embedding模型
# 前提：需提前通过ollama pull qwen3-embedding:4b 下载模型
model = OllamaEmbeddings(model="qwen3-embedding:4b")

# embed_query: 单句向量化
print(model.embed_query("我喜欢你"))

# embed_documents: 批量向量化
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))
