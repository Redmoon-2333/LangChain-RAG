"""
Agent工具集定义模块
包含7个工具函数：RAG检索、天气查询、用户信息获取、外部数据查询、上下文填充
"""

from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService
import random, os
from utils.config_handler import agent_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger


# 初始化RAG服务（全项目共用单例，避免重复初始化）
vector_store = VectorStoreService()
rag = RagSummarizeService(vector_store)

# 模拟用户数据
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

# 外部数据缓存（懒加载，只在首次调用时加载）
external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """RAG检索工具：将用户问题转换为向量，从知识库检索相关文档并生成摘要"""
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符形式返回")
def get_weather(city: str) -> str:
    """天气查询工具：返回城市的天气信息（模拟数据）"""
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时内降雨概率极低"


@tool(description="获取用户所在城市名称，以纯字符形式返回")
def get_user_location() -> str:
    """用户定位工具：返回当前用户所在城市（随机模拟）"""
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户ID，以纯字符形式返回")
def get_user_id() -> str:
    """用户ID工具：返回当前用户ID（随机模拟）"""
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符形式返回")
def get_current_month() -> str:
    """时间获取工具：返回当前月份（随机模拟）"""
    return random.choice(month_arr)


def generate_external_data():
    """
    从CSV文件加载外部用户数据（懒加载模式）
    Warning: 仅在首次调用时加载，大文件需要注意内存占用
    """
    # 避免重复加载
    if not external_data:
        if "external_data_path" not in agent_conf:
            raise KeyError("配置中缺少 external_data_path 字段")

        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")

        # CSV格式：user_id,feature,efficiency,consumables,comparison,time
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过表头
                arr = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                # 按用户ID和时间构建嵌套字典
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="检索指定用户在指定月份的扫地/扫拖机器人完整使用记录，以纯字符形式返回，如未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部数据查询工具：根据用户ID和月份查询使用记录
    :param user_id: 用户ID字符串
    :param month: 月份字符串，格式如"2025-01"
    :return: 使用记录字典的字符串形式，未找到返回空字符串
    """
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warn(f"[fetch_external_data]未能检索到用户:{user_id}在{month}的数据。")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    """
    上下文填充工具：标记当前会话进入"报告生成模式"
    触发机制：Agent调用此工具后，中间件检测到工具名，动态修改runtime.context["report"]=True
    应用场景：当用户请求生成报告时调用，触发提示词切换
    """
    return "fill_context_for_report已调用"
