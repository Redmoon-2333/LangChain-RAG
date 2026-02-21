"""
TextLoader和文档分割器：加载文本文件并按规则分割成小块
核心：RecursiveCharacterTextSplitter支持多种分隔符，递归分割保证语义完整
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文本文件
loader = TextLoader("./data/Python基础语法.txt", encoding="utf-8")
docs = loader.load()

# 文档分割器：递归分割文档
# 原理：按分隔符列表顺序尝试分割，遇到能分割的位置就停下，保证语义完整
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 分段的最大字符数
    chunk_overlap=50,      # 分段之间允许重叠字符数（保持语义连贯）
    # 文本自然段落分隔的依据符号：按优先级排序
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    length_function=len,   # 统计字符的依据函数
)

split_docs = splitter.split_documents(docs)
print(len(split_docs))
for doc in split_docs:
    print("="*20)
    print(doc)
    print("="*20)
