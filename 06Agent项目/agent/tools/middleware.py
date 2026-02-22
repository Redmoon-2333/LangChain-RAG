"""
Agent中间件模块
实现3个中间件：工具执行监控、模型调用日志、动态提示词切换
"""

from typing import Callable, Any
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt, load_report_prompt


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """
    工具执行监控中间件
    功能：
    1. 记录工具调用日志（工具名、参数、结果）
    2. 特殊处理fill_context_for_report工具，动态修改runtime.context
    """
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]参数: {request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # 特殊逻辑：检测到fill_context_for_report被调用时，设置报告模式标志
        if request.tool_call['name'] == 'fill_context_for_report':
            logger.info(f"[tool monitor]fill_context_for_report工具被调用，注入上下文 report=True")
            # 通过runtime.context在中间件链中传递状态
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.info(f"工具{request.tool_call['name']}调用失败: {e}")
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    模型调用前日志中间件
    功能：记录即将调用LLM时的状态信息（消息数量、最新消息内容）
    """
    logger.info(f"[log_before_model]: 即将调用模型，带有{len(state['messages'])}条消息，消息如下：")
    # for message in state['messages']:
    #     logger.info(f"[log_before_model][{type(message).__name__}]: {message.content.strip()}")
    logger.info(f"[log_before_model]: ----------省略已输出内容----------")
    # 仅打印最新一条消息，避免日志过长
    logger.info(f"[log_before_model][{type(state['messages'][-1]).__name__}]: {state['messages'][-1].content.strip()}")

    # 返回None表示不修改prompt，继续正常流程
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    """
    动态提示词切换中间件
    功能：根据runtime.context["report"]标志位，动态切换系统提示词
    - report=False: 使用main_prompt（普通客服模式）
    - report=True: 使用report_prompt（报告生成模式）
    原理：dynamic_prompt中间件在每次模型调用前被触发，允许动态修改提示词
    """
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()

    return load_system_prompt()
