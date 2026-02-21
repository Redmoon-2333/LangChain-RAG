"""
JSONLoader使用：从JSON文件加载文档数据
核心：使用jq_schema语法提取JSON中的特定字段，支持标准JSON和JSONLines格式
"""

from langchain_community.document_loaders import JSONLoader

# 方式1：加载JSON数组，提取所有元素的name字段
loader = JSONLoader(
    file_path="./data/stus.json",
    jq_schema=".[].name",      # ".[].name" 提取数组中每个元素的name字段
    text_content=False,         # 告知JSONLoader抽取的内容不是字符串（是字段值）
    json_lines=False           # 告知JSONLoader这是标准JSON文件（非JSONLines）
)

document = loader.load()
print(document)

# 需要安装jq组件

# 方式2：加载单个JSON对象，提取name字段
loader = JSONLoader(
    file_path="./data/stu.json",
    jq_schema=".name",         # ".name" 提取对象的name字段
    text_content=False,
    json_lines=False
)

document = loader.load()
print(document)

# 方式3：加载JSONLines文件（每行一个独立JSON）
loader = JSONLoader(
    file_path="./data/stu_json_lines.json",
    jq_schema=".name",
    text_content=False,
    json_lines=True            # 告知JSONLoader这是JSONLines文件
)

document = loader.load()
print(document)
