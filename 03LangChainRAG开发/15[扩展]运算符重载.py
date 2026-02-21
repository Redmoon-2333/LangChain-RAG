"""
Python运算符重载：深入理解|在LangChain中的工作原理
核心：通过__or__方法实现|运算符的自定义行为，理解LCEL链式调用的底层机制
"""

class Test(object):
    def __init__(self, name):
        self.name = name

    # 实现|运算符重载：当执行 a | b 时，调用 a.__or__(b)
    def __or__(self, other):
        return MySequence(self, other)  # 首次串联创建MySequence

    def __str__(self):
        return self.name


class MySequence(object):
    def __init__(self, *args):
        self.sequence = []
        for arg in args:
            self.sequence.append(arg)

    # 继续重载|，实现链式调用
    # a | b | c 实际执行：a.__or__(b).__or__(c)
    def __or__(self, other):
        self.sequence.append(other)
        return self  # 返回self支持连续|调用

    def run(self):
        for i in self.sequence:
            print(i)


if __name__ == '__main__':
    a = Test('a')
    b = Test('b')
    c = Test('c')
    e = Test('e')
    f = Test('f')
    g = Test('g')

    # a | b | c | e | f | g 等价于 a.__or__(b).__or__(c).__or__(e).__or__(f).__or__(g)
    d = a | b | c | e | f | g
    d.run()
    print(type(d))
