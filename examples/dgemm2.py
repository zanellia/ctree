from __future__ import print_function
from ctree import get_ast, browser_show_ast
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
import numpy as np
from ctree.transformations import PyBasicConversions

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ConcreteDgemm(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_typesig)

    def __call__(self, a, b, c):
        return self._c_function(a, b, c)


class Dgemm(LazySpecializedFunction):
    def __init__(self):
        tree = get_ast(Dgemm.multiply)
        super(Dgemm, self).__init__(tree)

    @staticmethod
    def multiply(a, b, c=None, alpha=1.0, beta=0.0):
        n = a.shape[0]
        m = a.shape[1]
        p = b.shape[1]

        assert m == b.shape[0]
        if c is None:
            c = np.zeros((a.shape[0], b.shape[1]))
        else:
            assert c.shape == (n, p)

        for i in range(n):
            for j in range(p):
                c[(i,j)] *= beta
                for k in range(m):
                    c[(i, j)] += alpha * a[(i, k)] * b[(k, j)]
        return c

    def transform(self, tree, program_config):
        transforms = [
            PyBasicConversions()
        ]

        for t in transforms:
            tree = t.visit(tree)

        browser_show_ast(tree)



if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 3, 5, 8], [2, 4, 6, 8], [3, 6, 9, 12]])

    print("a.shape {} b.shape {}".format(a.shape, b.shape))

    c = Dgemm.multiply(a, b)

    print("c\n{}".format(c))

    dgemm = Dgemm()

    c = dgemm(a, b)

    print("c\n{}".format(c))
