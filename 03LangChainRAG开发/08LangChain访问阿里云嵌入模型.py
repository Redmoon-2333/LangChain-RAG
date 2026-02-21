"""
阿里云嵌入模型：文本向量化（Text Embedding）
作用：将文本转换为固定维度的向量，用于语义搜索、相似度计算等场景
核心：DashScopeEmbeddings默认使用text-embeddings-v1模型，向量维度1536
"""

from langchain_community.embeddings import DashScopeEmbeddings

# 初始化嵌入模型：默认 text-embeddings-v1，向量维度1536
# Warning: 嵌入模型的选择直接影响向量检索效果，需根据任务选择合适模型
model = DashScopeEmbeddings()

# embed_query: 单句向量化，将文本转换为向量
# 输入：str类型，返回：list[float]，维度1536
print(model.embed_query("我喜欢你"))

# embed_documents: 批量向量化，一次处理多条文本
# 输入：list[str]，返回：list[list[float]]
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))
