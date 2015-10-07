__author__ = 'dorthyluu'

import unittest
import ast
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import Array

class TestList(unittest.TestCase):

    def test_List(self):
        array = ast.parse("[1, 5, 7, 3]")
        array = PyBasicConversions().visit(array).find(Array)
        self.assertEqual(str(array), "{1, 5, 7, 3}")
