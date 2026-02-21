# langchain_community: LangChain社区提供的第三方集成包
# 引入通义千问大模型（阿里云）
from langchain_community.llms.tongyi import Tongyi

# 初始化模型：model参数指定使用的通义模型
# Warning: 生产环境应使用环境变量存储API Key，避免硬编码
model = Tongyi(model="qwen-max")

# invoke: 同步调用方法，返回完整响应（非流式）
# 参数input: str类型，表示向模型发送的提示词
res = model.invoke(input="你是谁呀能做什么？")

print(res)
