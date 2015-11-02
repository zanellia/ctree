from __future__ import print_function
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
import numpy as np

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ConcreteDgemm(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_typesig)

    def __call__(self, a, b, c):
        return self._c_function(a, b, c)


class Dgemm(LazySpecializedFunction):
    def multiply(self, a, b, c, alpha=1.0, beta=0.0):
        n = a.shape[0]
        m = a.shape[1]
        p = b.shape[1]

        assert m == b.shape[0]
        assert c.shape == (n, p)

        for i in range(n):
            for j in range(p):
                c[(i,j)] *= beta
                for k in range(m):
                    c[(i, j)] += alpha * a[(i, k)] * b[(k, j)]


if __name__ == '__main__':
    dgemm = Dgemm()

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 3, 5, 8], [2, 4, 6, 8], [3, 6, 9, 12]])

    print("a.shape {} b.shape {}".format(a.shape, b.shape))

    c = np.zeros((a.shape[0], b.shape[1]))

    dgemm.multiply(a, b, c)

    print("c\n{}".format(c))
