from collections import namedtuple


def convert(dictionary):
    # 说明: 将字典转换为支持属性访问的 namedtuple
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
