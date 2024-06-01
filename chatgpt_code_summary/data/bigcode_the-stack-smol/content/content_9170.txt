#!/usr/bin/env python
# encoding: utf-8


class MeanVariance(object):
    def __init__(self):
        self.n = 0
        self.K = 0.0
        self.ex0 = 0.0
        self.ex2 = 0.0

    def add_variable(self, x):
        if self.n == 0:
            self.K = x
        self.n += 1
        delta = x - self.K
        self.ex0 += delta
        self.ex2 += delta * delta

    def get_mean_value(self, ):
        return self.K + self.ex0 / self.n

    def get_variance(self, ):
        return (self.ex2 - (self.ex0 * self.ex0) / self.n) / (self.n - 1)


if __name__ == "__main__":
    mv = MeanVariance()
    mv.add_variable(1.0)
    mv.add_variable(2.0)
    mv.add_variable(3.0)
    mv.add_variable(4.0)
    print mv.get_mean_value()
