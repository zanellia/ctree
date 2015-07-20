import unittest
import ast
from ctree.c.nodes import BinaryOp
from ctree.transformations import PyBasicConversions


class TestCompare(unittest.TestCase):

    def test_LessThan(self):
        comp = ast.parse("5 < foo < 6")
        comp = PyBasicConversions().visit(comp).find(BinaryOp)
        self.assertEqual(str(comp), "5 < foo && foo < 6")

    def test_LessThanEqual(self):
        comp = ast.parse("5 <= foo <= 6")
        comp = PyBasicConversions().visit(comp).find(BinaryOp)
        self.assertEqual(str(comp), "5 <= foo && foo <= 6")

    def test_GreaterThan(self):
        comp = ast.parse("5 > foo > 6")
        comp = PyBasicConversions().visit(comp).find(BinaryOp)
        self.assertEqual(str(comp), "5 > foo && foo > 6")

    def test_GreaterThan(self):
        comp = ast.parse("5 >= foo >= 6")
        comp = PyBasicConversions().visit(comp).find(BinaryOp)
        self.assertEqual(str(comp), "5 >= foo && foo >= 6")

    def test_Equals(self):
        comp = ast.parse("5 == foo == 6")
        comp = PyBasicConversions().visit(comp).find(BinaryOp)
        self.assertEqual(str(comp), "5 == foo && foo == 6")