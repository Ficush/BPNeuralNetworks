# -*- coding: utf-8 -*-

import BPANN

# Demo Function
def main():

    # 测试数据
    data = [
        [[0,0], [0,0,0]],
        [[0,0.5], [0,1,0]],
        [[0.5,0], [1,0,0]],
        [[1,1], [1,1,0]],
        [[1,1], [1,1,0]],
    ]

    # 实例化神经网络类
    ann = BPANN.BPNeuralNetworks(2, 5, 3)
    ann.train(data, iterations = 10000)
    ann.caclulate(data)

if __name__ == '__main__':
    main()