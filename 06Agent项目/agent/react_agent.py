"""
ReAct Agent 智能客服机器人
核心入口：封装ReactAgent类，提供流式对话能力
"""

from langchain.agents import create_agent
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                                     fetch_external_data, fill_context_for_report)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt


class ReactAgent(object):
    """
    智能客服Agent封装类
    整合LLM、工具列表、中间件，提供流式输出能力
    """

    def __init__(self):
        """
        初始化Agent实例
        - model: 阿里云通义千问作为推理引擎
        - system_prompt: 从外部文件加载，定义Agent角色
        - tools: 7个工具（检索、天气、用户信息、外部数据、报告生成）
        - middleware: 3个中间件（工具监控、模型调用日志、动态提示词）
        """
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query):
        """
        流式执行用户查询
        :param query: 用户问题字符串
        :return: Generator[str]，逐块返回Agent响应
        """
        # 构建输入字典，兼容LangChain的消息格式
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # context参数：通过runtime.context在中间件间传递状态
        # report=False 表示普通对话模式，True表示报告生成模式
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]  # 有历史记录所以取最后一条
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


# 单元测试入口
if __name__ == '__main__':
    agent = ReactAgent()
    # 测试：结合用户地理位置和天气的场景
    for chunk in agent.execute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(chunk, end="", flush=True)
