"""
LangChain Agent 流式输出示例
演示如何使用 stream 方法实时获取 Agent 的推理过程和中间状态
"""

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="获取股价，传入股票名称，返回字符串信息")
def get_price(name: str) -> str:
    """单参数工具：根据股票名称查询价格"""
    return f"股票{name}的价格是20元"


@tool(description="获取股票信息，传入股票名称，返回字符串信息")
def get_info(name: str) -> str:
    """单参数工具：根据股票名称查询公司信息"""
    return f"股票{name}，是一家A股上市公司，专注于IT职业教育。"


# 创建 Agent，配置多个工具
agent = create_agent(
    model=ChatTongyi(model="qwen3-max-2026-01-23"),
    tools=[get_price, get_info],
    system_prompt="你是一个智能助手，可以回答股票相关问题，记住请告知我思考过程，让我知道你为什么调用某个工具"
)

# stream: 流式调用，返回 Generator
# - stream_mode="values": 返回每一步的完整状态快照
# 与 invoke 的区别：
#   - invoke: 等待完整结果返回
#   - stream: 实时yield中间状态，便于观察 Agent 推理过程
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "传智教育股价多少，并介绍一下"}]},
    stream_mode="values"
):
    latest_message = chunk['messages'][-1]

    # chunk["messages"]: 每个流式块包含完整的消息历史
    # latest_message: 当前最新的消息（可能是思考、工具调用或最终回复）
    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)

    # tool_calls: Agent 决定调用工具时，会在此字段记录
    # 格式: [{"name": "工具名", "args": {"参数": "值"}, "id": "调用ID"}]
    try:
        if latest_message.tool_calls:
            print(f"工具调用： { [tc['name'] for tc in latest_message.tool_calls]  }")
    except AttributeError as e:
        pass
