import unittest

from ctree.c.macros import *
from ctree.cpp.dotgen import CppDotLabeller
from ctree.cpp.nodes import CppInclude, CppComment


class TestCMacrosCodegen(unittest.TestCase):
    def test_null(self):
        node = NULL()
        self.assertEqual(str(node), "NULL")

    def test_printf(self):
        node = printf("%s %s", SymbolRef("x"), SymbolRef("y"))
        self.assertEqual(str(node), "printf(\"%s %s\", x, y)")


class TestCppMacrosCodegen(unittest.TestCase):
    def test_include(self):
        labeller = CppDotLabeller()

        cpn = CppInclude(target="dog")
        result = labeller.visit(cpn)
        self.assertRegexpMatches(result, r"target: <dog>")

        cpn = CppInclude(target="dog", angled_brackets=False)
        result = labeller.visit(cpn)
        self.assertRegexpMatches(result, r'target: "dog"')

    def test_comment(self):
        labeller = CppDotLabeller()

        cpn = CppComment(text="fox")
        result = labeller.visit(cpn)
        self.assertRegexpMatches(result, r"// fox")

        # quotes are escaped in comment
        cpn = CppComment(text='"this" is quoted')
        result = labeller.visit(cpn)
        self.assertRegexpMatches(result, r'// \\"this\\" is quoted')
