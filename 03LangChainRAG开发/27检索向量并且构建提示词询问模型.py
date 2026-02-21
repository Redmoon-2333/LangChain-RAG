"""
RAG基础流程：检索向量库并构建提示词询问模型
核心：先通过向量检索获取相关资料，再将资料作为上下文提供给模型回答问题
"""

from langchain_community.chat_models import ChatTongyi
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

# 构建提示词模板：系统消息包含参考资料
prompt = ChatPromptTemplate.from_messages(
    [
        # system消息：要求模型基于提供的参考资料回答问题
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问：{input}")
    ]
)

# 初始化向量存储
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings())

# 准备知识库数据：添加文本到向量库
# add_texts: 传入文本列表，自动进行向量化
vector_store.add_texts(
    ["减肥就是要少吃多练", "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"]
)

input_text = "怎么减肥？"

# Step1: 检索向量库 - 根据用户问题查找相似文档
result = vector_store.similarity_search(input_text, 2)

# Step2: 拼接参考资料
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

# Step3: 构建完整提示词并调用模型
chain = prompt | print_prompt | model | StrOutputParser()

res = chain.invoke({"input": input_text, "context": reference_text})
print(res)
