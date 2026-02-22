"""
Streamlit Web应用入口
智扫通机器人智能客服系统
功能：基于ReAct Agent的智能客服聊天界面，支持流式输出
"""

import time
import streamlit as st
from agent.react_agent import ReactAgent

# 页面标题
st.title("智扫通机器人智能客服")
st.divider()

# 初始化会话状态：消息历史
# Warning: Streamlit重载页面时session_state会重置，需持久化存储请使用外部数据库
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？"}]

# 初始化会话状态：Agent实例（避免每次重载重复创建）
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 渲染消息历史
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 存储完整响应（用于保存到历史）
    response_messages = []

    # 显示加载动画
    with st.spinner("智能客服思考中..."):
        # 调用Agent流式输出
        res_stream = st.session_state["agent"].execute_stream(prompt)

        # 流式字符渲染器：控制输出速度，模拟打字机效果
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)  # 缓存完整响应

                for char in chunk:
                    time.sleep(0.01)  # 控制每个字符的显示间隔
                    yield char

        # 实时显示AI回复
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))

        # 保存AI回复到会话历史
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})

        # 刷新页面（重新渲染消息历史）
        st.rerun()
