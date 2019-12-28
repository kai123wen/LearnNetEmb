import sys
from os import path


def t():
    dirname = path.dirname(path.abspath(__file__))
    print(sys.platform)
    print(path.join(dirname, r'..\windows\liblinear.dll'))


if __name__ == "__main__":
    t()
