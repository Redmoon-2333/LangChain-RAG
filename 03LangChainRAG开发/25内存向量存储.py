"""
内存向量存储(InMemoryVectorStore)：将文档向量存储在内存中
特点：进程结束后数据丢失，适用于小数据集和开发测试
核心：add_documents添加文档，similarity_search执行语义检索
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# 初始化向量存储：指定嵌入模型
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings()
)


loader = CSVLoader(
    file_path="./data/info.csv",
    encoding="utf-8",
    source_column="source",     # 指定本条数据的来源字段
)

documents = loader.load()

# 添加文档到向量存储
# ids参数：手动指定文档ID，便于后续删除和检索
vector_store.add_documents(
    documents=documents,        # 被添加的文档，类型：list[Document]
    ids=["id"+str(i) for i in range(1, len(documents)+1)]  # 给添加的文档提供id
)

# 删除文档：根据ID删除
vector_store.delete(["id1", "id2"])

# 语义检索：根据输入查询，返回最相似的文档
# 参数2：返回结果数量(k)
result = vector_store.similarity_search(
    "Python简单易学吗",
    3,       # 检索的结果要几个
)

print(result)
