# -*- coding: utf-8 -*-
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class Chainer(Chain):

    def __init__(self):
        super(Chainer, self).__init__(
            l1 = L.Linear(3, 100),
            l2 = L.Linear(100, 10),
            l3 = L.Linear(10, 1)
        )

    def predict(self, x):
        h1 = F.identity(self.l1(x))
        h2 = F.identity(self.l2(h1))
        return self.l3(h2)
