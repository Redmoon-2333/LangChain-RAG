"""
RunnablePassthrough使用：实现RAG流程的Chain整合
核心：允许输入直接传递到下一步，结合retriever实现自动检索和提示词构建
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def print_prompt(prompt):
    """调试函数：打印最终生成的prompt"""
    print(prompt.to_string())
    print("=" * 20)
    return prompt


model = ChatTongyi(model="qwen3-max-2026-01-23")

# 构建提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问：{input}")
    ]
)

# 初始化向量存储并添加数据
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings())
vector_store.add_texts(
    ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"]
)

input_text = "怎么减肥？"

# retriever: 向量存储转换为检索器
# as_retriever返回Runnable接口的实例，可直接接入Chain
# search_kwargs: 检索参数，k指定返回结果数量
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def format_func(docs: list[Document]):
    """格式化检索结果：将Document列表转换为字符串"""
    if not docs:
        return "无相关参考资料"

    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"

    return formatted_str


# Chain整合：输入 → retriever检索 → format_func格式化 → prompt构建 → model → parser
# 数据流向：
#   - RunnablePassthrough(): 将input_text原样传递给prompt的input字段
#   - retriever | format_func: 将检索结果格式化后传递给prompt的context字段
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
)

res = chain.invoke(input_text)
print(res)
