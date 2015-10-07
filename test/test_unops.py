import unittest

from ctree.c.nodes import *
from ctree.transformations import PyBasicConversions
import ast


class TestUnaryOps(unittest.TestCase):
    def setUp(self):
        self.foo = SymbolRef("foo")

    def _check(self, op_cls, expected_string):
        node = op_cls(self.foo)
        self.assertEqual(str(node), expected_string)

    def test_plus(self):
        self._check(Add, "+ foo")

    def test_minus(self):
        self._check(Sub, "- foo")

    def test_bitnot(self):
        self._check(BitNot, "~ foo")

    def test_not(self):
        self._check(Not, "! foo")

    def test_ref(self):
        self._check(Ref, "& foo")

    def test_deref(self):
        self._check(Deref, "* foo")

    def test_preinc(self):
        self._check(PreInc, "++ foo")

    def test_predec(self):
        self._check(PreDec, "-- foo")

    def test_postinc(self):
        self._check(PostInc, "foo ++")

    def test_postdec(self):
        self._check(PostDec, "foo --")

    def test_sizeof(self):
        self._check(SizeOf, "sizeof foo")

class TestPyBasicConversionsUnaryOps(unittest.TestCase):
    def _check(self, op, expected_string):
        self.assertEqual(str(op), expected_string)

    def test_plus(self):
        op = ast.parse("+ foo")
        op = PyBasicConversions().visit(op).find(UnaryOp)
        self._check(op, "+ foo")

    def test_minus(self):
        op = ast.parse("- foo")
        op = PyBasicConversions().visit(op).find(UnaryOp)
        self._check(op, "- foo")

    def test_bitnot(self):
        op = ast.parse("~ foo")
        op = PyBasicConversions().visit(op).find(UnaryOp)
        self._check(op, "~ foo")

    def test_not(self):
        op = ast.parse("not foo")
        op = PyBasicConversions().visit(op).find(UnaryOp)
        self._check(op, "! foo")

    def test_CUnaryOp(self):
        op = Not(SymbolRef("foo"))
        op = PyBasicConversions().visit(op).find(UnaryOp)
        self._check(str(op), "! foo")
