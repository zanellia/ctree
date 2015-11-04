import unittest
from ctree.cpp.codegen import CppCodeGen
from ctree.cpp.dotgen import CppDotLabeller

from ctree.cpp.nodes import *
from ctree.c.nodes import Add, SymbolRef, Constant


class TestCppIncludes(unittest.TestCase):
    def test_include_angled(self):
        tree = CppInclude("foo.h")
        self.assertEqual(str(tree), "#include <foo.h>")

        i0 = CppInclude("foo.h", angled_brackets=True)
        i1 = CppInclude("foo.h", angled_brackets=False)

        self.assertEqual(hash(tree), hash(i0))
        self.assertNotEqual(hash(tree), hash(i1))

    def test_include_quotes(self):
        tree = CppInclude("foo.h", angled_brackets=False)
        self.assertEqual(str(tree), '#include "foo.h"')


class TestDefines(unittest.TestCase):
    def _check(self, node, expected_string):
        self.assertEqual(str(node), expected_string)

    def test_simple_macro(self):
        d1, d2 = SymbolRef("d1"), SymbolRef("d2")
        node = CppDefine("test_macro", [d1, d2], Add(d1, d2))
        self._check(node, "#define test_macro(d1, d2) (d1 + d2)")

    def test_no_args(self):
        node = CppDefine("test_macro", [], Constant(39))
        self._check(node, "#define test_macro() (39)")


class TestCppBasics(unittest.TestCase):
    def test_cpp_basics(self):
        cpn = CppNode()

        self.assertEqual(cpn.label(), CppDotLabeller().visit(cpn))

        self.assertFalse(cpn._requires_semicolon())

    def test_cpp_code_gen(self):
        comment = CppComment("dragon")

        self.assertRegexpMatches(CppCodeGen().visit(comment), "// dragon")
