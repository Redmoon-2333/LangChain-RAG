"""
CSVLoader使用：从CSV文件加载文档数据
核心：每行数据作为一个Document，source字段记录数据来源
"""

from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(
    file_path="./data/stu.csv",
    csv_args={
        "delimiter": ",",       # 指定分隔符，默认逗号
        "quotechar": '"',       # 指定带有分隔符文本的引号包围是单引号还是双引号
        # 如果数据原本有表头，就不要下面的代码，如果没有可以使用
        # "fieldnames": ['name', 'age', 'gender', '爱好']
    },
    encoding="utf-8"            # 指定编码为UTF-8
)

# 批量加载：一次性读取所有文档，返回Document列表
documents = loader.load()

for document in documents:
    print(type(document),"\n||||||\n", document,'\n')

print ("======" * 20)
# 懒加载：按需加载，节省内存，适用于大数据集
# lazy_load返回迭代器(Iterator)，每次迭代返回一个Document
for document in loader.lazy_load():
    print(document,'\n')
