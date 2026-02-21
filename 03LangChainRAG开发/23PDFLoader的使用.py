"""
PDFLoader使用：从PDF文件加载文档数据
核心：支持单页模式和单文档模式，可设置密码保护
依赖：pip install pypdf
"""

from langchain_community.document_loaders import PyPDFLoader

# 方式1：single模式 - 无论有多少页，只返回1个Document对象
loader = PyPDFLoader(
    file_path="./data/pdf2.pdf",
    mode="single",        # single模式：整个PDF作为一个Document
    password="itheima"   # PDF密码（如有）
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("="*20, i)


# 方式2：page模式（默认）- 每个页面形成一个Document对象
loader = PyPDFLoader(
    file_path="./data/pdf1.pdf",
    mode="page",        # 默认模式：每个页面一个Document
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("="*20, i)