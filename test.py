from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from Chainer import Chainer
import numpy as np

from chainer import serializers
# まず同じネットワークのオブジェクトを作る
model = Chainer()

# そのオブジェクトに保存済みパラメータをロードする
serializers.load_npz('chainer2.model', model)

num1 = input("num1: ")
num2 = input("num2: ")
num3 = input("num3: ")
num1 = int(num1)
num2 = int(num2)
num3 = int(num3)

x = [ [ num1, num2, num3 ] ]
x = Variable(np.array(x, dtype=np.float32))
print(x)
t = model.predict(x)
print(t)
