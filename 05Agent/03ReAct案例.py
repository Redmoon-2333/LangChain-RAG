"""
LangChain ReAct Agent 示例
演示如何让 Agent 遵循 ReAct (Reasoning + Acting) 思维框架
"""

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="获取体重，返回值是整数，单位千克")
def get_weight() -> int:
    """无参数工具：获取用户体重"""
    return 90


@tool(description="获取身高，返回值是整数，单位厘米")
def get_height() -> int:
    """无参数工具：获取用户身高"""
    return 172


# ReAct 框架核心：通过 system_prompt 约束 Agent 的行为模式
# ReAct = Reasoning (推理) + Acting (行动)
# 思维流程：思考 → 行动 → 观察 → 再思考 → ... → 最终答案
# 关键约束：每轮仅调用一个工具，避免并行调用导致无法追踪观察结果
agent = create_agent(
    model=ChatTongyi(model="qwen3-max-2026-01-23"),
    tools=[get_weight, get_height],
    system_prompt="""你是严格遵循ReAct框架的智能体，必须按「思考→行动→观察→再思考」的流程解决问题，
    且**每轮仅能思考并调用1个工具**，禁止单次调用多个工具。
    并告知我你的思考过程，工具的调用原因，按思考、行动、观察三个结构告知我""",
)

# 使用 stream 模式观察 Agent 的 ReAct 推理过程
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "计算我的BMI"}]},
    stream_mode="values"
):
    latest_message = chunk['messages'][-1]

    # 输出消息内容（思考过程、工具调用结果等）
    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)

    # 捕获工具调用信息，观察 Agent 的行动决策
    try:
        if latest_message.tool_calls:
            print(f"工具调用： { [tc['name'] for tc in latest_message.tool_calls]  }")
    except AttributeError as e:
        pass
