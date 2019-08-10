# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from Chainer import Chainer

def data_read( ft_name, fa_name ):
    teachers = np.array([] )
    answers =  np.array([] )

    ft = open( ft_name, mode = "r" )
    fa = open( fa_name, mode = "r" )

    ft_data_string = ft.readlines()
    fa_data_string = fa.readlines()

    for i in range( 0, len( ft_data_string ) ):
        fa_data = fa_data_string[i].replace( "\n", "" )
        answers = np.append( answers, float( fa_data ) )
        
        ft_data = ft_data_string[i].replace( "\n", "" )
        ft_data = ft_data.split( " " )

        for r in range( 0, len( ft_data ) ):
            teachers = np.append( teachers ,float( ft_data[r] ) )

    ft.close()
    fa.close()

    teachers = teachers.astype( np.float32 )
    answers = answers.astype( np.float32 )

    teachers = np.reshape( teachers, ( int( len( teachers ) / 3 ), 3 ) )
    answers = np.reshape( answers, ( len( answers ) , 1 ) )

    return teachers, answers

# 損失関数の計算
# 損失関数には自乗誤差(MSE)を使用
def forward(x, y, model):
    t = model.predict(x)
    loss = F.mean_squared_error(t, y)
    return loss

def plot(x):
    # プロット
    t = model.predict(x)
    plt.plot(t.data, y.data)
    plt.scatter(t.data, t.data)
    plt.grid(which='major',color='gray',linestyle='-')
    plt.ylim(0, 300)
    plt.xlim(0, 300)
    plt.savefig("../aaa.png")


if __name__ == '__main__':
    # 乱数のシードを固定
    random.seed(1)

    # 標本データの生成
    x, y = data_read( "teacher2.txt", "answer2.txt" )

    # chainerの変数として再度宣言
    x = Variable(np.array(x, dtype=np.float32))
    y = Variable(np.array(y, dtype=np.float32))
    # NNモデルを宣言
    model = Chainer()

    # chainerのoptimizer
    #最適化のアルゴリズムには ADAM を使用
    optimizer = optimizers.Adam()
    # modelのパラメータをoptimizerに渡す
    optimizer.setup(model)
    
    # パラメータの学習を繰り返す
    for i in range(0,1000):
        loss = forward(x, y, model)
        print(loss.data)  # 現状のMSEを表示
        optimizer.update(forward, x, y, model)
    print(x)
    plot(x)

    from chainer import serializers
    serializers.save_npz('chainer2.model', model)
