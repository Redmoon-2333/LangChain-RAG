"""
LangChain Agent 中间件示例
演示如何通过中间件拦截 Agent 执行过程中的各个阶段，实现监控、日志、修改行为等功能
"""

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.runtime import Runtime


@tool(description="查询天气，传入城市名称字符串，返回字符串天气信息")
def get_weather(city: str) -> str:
    """单参数工具：根据城市名称查询天气"""
    return f"{city}天气：晴天"


# ============================================================
# 中间件类型说明：
# 1. before_agent: agent执行前触发，可用于日志记录、状态初始化
# 2. after_agent: agent执行后触发，可用于结果处理、资源清理
# 3. before_model: 模型调用前触发，可在调用前修改输入
# 4. after_model: 模型调用后触发，可在调用后修改输出
# 5. wrap_model_call: 包装模型调用，可添加前后处理逻辑
# 6. wrap_tool_call: 包装工具调用，可监控工具执行、修改参数
# ============================================================


# before_agent: Agent 执行前调用
# 参数 state: 包含 messages 等状态数据的字典
# 参数 runtime: 执行时上下文，可用于控制流程、获取配置等
@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    # agent执行前会调用这个函数并传入state和runtime两个对象
    print(f"[before agent]agent启动，并附带{len(state['messages'])}消息")


# after_agent: Agent 执行完成后调用
@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after agent]agent结束，并附带{len(state['messages'])}消息")


# before_model: LLM 模型调用前触发
# 此时可以修改 prompt 或注入额外上下文
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[before_model]模型即将调用，并附带{len(state['messages'])}消息")


# after_model: LLM 模型调用后触发
# 此时可以检查/修改模型输出，或记录日志
@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_model]模型调用结束，并附带{len(state['messages'])}消息")


# wrap_model_call: 函数装饰器模式包装模型调用
# - request: 模型输入请求对象
# - handler: 实际执行模型的回调函数
# 典型用途：添加重试逻辑、计时、缓存等
@wrap_model_call
def model_call_hook(request, handler):
    print("模型调用啦")
    return handler(request)


# wrap_tool_call: 函数装饰器模式包装工具调用
# - request: 工具调用请求，包含 tool_call 字段（工具名、参数等）
# - handler: 实际执行工具的回调函数
# 典型用途：参数校验、执行监控、结果缓存等
@wrap_tool_call
def monitor_tool(request, handler):
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数：{request.tool_call['args']}")

    return handler(request)


# 创建 Agent 并注册中间件
# middleware 参数接收中间件函数列表，按注册顺序依次执行
agent = create_agent(
    model=ChatTongyi(model="qwen3-max-2026-01-23"),
    tools=[get_weather],
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool]
)

# 调用 Agent，观察中间件输出
res = agent.invoke({"messages": [{"role": "user", "content": "深圳今天的天气如何呀，如何穿衣"}]})
print("**********\n", res)
