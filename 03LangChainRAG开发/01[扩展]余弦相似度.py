"""
文件功能：手动实现余弦相似度计算（为后续RAG向量检索打基础）
"""

import numpy as np


def get_dot(vec_a, vec_b):
    """点积：同维度数字乘积之和"""
    if len(vec_a) != len(vec_b):
        raise ValueError("2个向量必须维度数量相同")

    dot_sum = 0
    for a, b in zip(vec_a, vec_b):
        dot_sum += a * b

    return dot_sum


def get_norm(vec):
    """模长：向量每个数字平方求和再开根号"""
    sum_square = 0
    for v in vec:
        sum_square += v * v

    return np.sqrt(sum_square)


def cosine_similarity(vec_a, vec_b):
    """余弦相似度：点积 / (模长A * 模长B)，范围[-1, 1]
    1: 方向相同 | 0: 正交 | -1: 方向相反
    """
    result = get_dot(vec_a, vec_b) / (get_norm(vec_a) * get_norm(vec_b))
    return result


if __name__ == '__main__':
    # 测试向量
    vec_a = [0.5, 0.5]
    vec_b = [0.7, 0.7]
    vec_c = [0.7, 0.5]
    vec_d = [-0.6, -0.5]

    print("ab:", cosine_similarity(vec_a, vec_b))  # 方向相近，接近1
    print("ac:", cosine_similarity(vec_a, vec_c))  # 方向略有差异
    print("ad:", cosine_similarity(vec_a, vec_d))  # 方向相反，负值
