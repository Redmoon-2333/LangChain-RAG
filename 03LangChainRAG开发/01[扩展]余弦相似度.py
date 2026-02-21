"""
手动实现余弦相似度计算：为后续RAG向量检索打基础
核心概念：
- 点积(Dot Product): 同维度向量对应元素相乘后求和
- 模长(Norm): 向量长度，元素平方和开根号
- 余弦相似度: 点积/(模长A*模长B)，范围[-1,1]
  1: 方向相同 | 0: 正交(无关) | -1: 方向相反
"""

import numpy as np


def get_dot(vec_a, vec_b):
    """点积：同维度向量对应元素相乘后求和
    
    几何意义：表示两个向量在方向上的相似程度
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("2个向量必须维度数量相同")

    dot_sum = 0
    for a, b in zip(vec_a, vec_b):
        dot_sum += a * b

    return dot_sum


def get_norm(vec):
    """模长(范数)：向量长度，元素平方和开根号
    
    L2范数：欧几里得距离
    """
    sum_square = 0
    for v in vec:
        sum_square += v * v

    return np.sqrt(sum_square)


def cosine_similarity(vec_a, vec_b):
    """余弦相似度：点积 / (模长A * 模长B)
    
    含义：向量夹角的余弦值，-1到1之间
    - 接近1：方向相近，语义相似
    - 接近0：方向正交，无相关性
    - 接近-1：方向相反，对立关系
    """
    result = get_dot(vec_a, vec_b) / (get_norm(vec_a) * get_norm(vec_b))
    return result


if __name__ == '__main__':
    # 测试向量
    vec_a = [0.5, 0.5]
    vec_b = [0.7, 0.7]
    vec_c = [0.7, 0.5]
    vec_d = [-0.6, -0.5]

    print("ab:", cosine_similarity(vec_a, vec_b))  # 方向完全相同，=1.0
    print("ac:", cosine_similarity(vec_a, vec_c))  # 方向略有差异，<1
    print("ad:", cosine_similarity(vec_a, vec_d))  # 方向相反，负值
