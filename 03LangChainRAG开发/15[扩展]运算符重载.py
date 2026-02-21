
class Test(object):
    def __init__(self, name):
        self.name = name

    def __or__(self, other):  # 实现|运算符重载
        return MySequence(self, other)  # 首次串联创建MySequence

    def __str__(self):
        return self.name


class MySequence(object):
    def __init__(self, *args):
        self.sequence = []
        for arg in args:
            self.sequence.append(arg)

    def __or__(self, other):  # 继续重载|，实现链式调用
        self.sequence.append(other)
        return self  # 返回self支持连续|

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

    d = a | b | c | e | f | g  # a.__or__(b).__or__(c)...
    d.run()
    print(type(d))
