"""
外部向量存储持久化：使用Chroma实现向量数据的持久化存储
核心：persist_directory指定存储路径，数据持久化保存，重启后仍可使用
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# Chroma向量数据库：轻量级本地向量数据库
# 依赖：pip install langchain-chroma chromadb
vector_store = Chroma(
    collection_name="test",     # 集合名称，类似数据库的表名
    embedding_function=DashScopeEmbeddings(),       # 嵌入模型
    persist_directory="./chroma_db"     # 指定数据持久化目录
)


loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source",     # 指定本条数据的来源字段
)

documents = loader.load()

# 添加文档到向量存储
vector_store.add_documents(
    documents=documents,        # 被添加的文档，类型：list[Document]
    ids=["id"+str(i) for i in range(1, len(documents)+1)]  # 文档ID
)

# # 删除文档：根据ID删除
# vector_store.delete(["id1", "id2"])

# 语义检索：支持filter参数过滤元数据
# filter: 字典形式，指定元数据过滤条件
result = vector_store.similarity_search(
    "Python是不是简单易学呀",
    3,        # 返回结果数量
    filter={"source": "黑马程序员"}  # 只返回source为"黑马程序员"的文档
)

print(result)
