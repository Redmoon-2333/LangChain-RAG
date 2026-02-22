"""
Chain调试工具模块
提供打印Prompt内容的辅助函数，用于调试LCEL Chain的输入输出
"""


def print_prompt(prompt, logger=None):
    """
    打印Prompt内容（调试用）
    :param prompt: Prompt对象（支持to_string()方法）
    :param logger: 可选，指定logger输出，默认print
    :return: 原prompt对象（透传）
    """
    if logger:
        logger.info(f"[print_prompt]" + "==========")
        logger.info(f"{prompt.to_string()}")
        logger.info(f"[print_prompt]" + "==========")
    else:
        print(f"[print_prompt]" + "==========")
        print(f"{prompt.to_string()}")
        print(f"[print_prompt]" + "==========")

    return prompt
