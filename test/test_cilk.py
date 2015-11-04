import unittest
from ctree.cilk.codegen import CilkCodeGen
from ctree.cilk.dotgen import CilkDotLabeller
from ctree.cilk.nodes import CilkNode


class TestCilk(unittest.TestCase):
    def test_cilk_codgen_visitor(self):
        self.assertIsInstance(CilkCodeGen(), CilkCodeGen)

    def test_cilk_dot_labeller(self):
        self.assertIsInstance(CilkDotLabeller(), CilkDotLabeller)

    def test_cilk_node(self):
        cilk_node = CilkNode()

        self.assertEqual(cilk_node.codegen(0), CilkCodeGen(0).visit(cilk_node))
        self.assertEqual(cilk_node._to_dot(0), CilkDotLabeller().visit(cilk_node))
