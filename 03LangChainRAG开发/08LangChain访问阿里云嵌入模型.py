"""
文件功能：阿里云嵌入模型（文本向量化）
"""

from langchain_community.embeddings import DashScopeEmbeddings

model = DashScopeEmbeddings()  # 默认 text-embeddings-v1，向量维度1536

# embed_query: 单句向量化，返回list[float]
print(model.embed_query("我喜欢你"))

# embed_documents: 批量向量化，返回list[list[float]]
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))
