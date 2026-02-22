"""
LangChain Agent 基础示例
演示如何使用 create_agent 创建最基础的 Agent 并调用工具
"""

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="查询天气")
def get_weather() -> str:
    """无参数工具：直接返回天气信息"""
    return "晴天"


# create_agent: LangChain 提供的 Agent 工厂函数
# 参数说明：
# - model: 指定 LLM 作为 Agent 的"大脑"，负责推理和决策
# - tools: 暴露给 Agent 的工具列表，Agent 可自主决定调用哪些工具
# - system_prompt: 系统提示词，定义 Agent 的角色和行为规则
agent = create_agent(
    model=ChatTongyi(model="qwen3-max-2026-01-23"),        # 智能体的大脑LLM
    tools=[get_weather],            # 向智能体提供工具列表
    system_prompt="你是一个聊天助手，可以回答用户问题。",
)

# invoke: 同步调用 Agent，输入 messages 格式与 OpenAI API 兼容
res = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "明天深圳的天气如何？"},
        ]
    }
)

# res["messages"]: Agent 完整的对话历史，包含用户消息、Agent 思考、工具调用结果等
for msg in res["messages"]:
    print(type(msg).__name__, msg.content)
