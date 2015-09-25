import numpy as np
import ctypes as ct
import ast

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.transformations import PyBasicConversions
from ctree.transforms.declaration_filler import DeclarationFiller
from ctree.c.nodes import CFile
import ctree.c.nodes as C
from ctree.nodes import Project
from ctree.types import get_ctype
from ctree.templates.nodes import StringTemplate


def get_nd_pointer(arg):
    return np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)


class HwachaFN(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_point_name, project_node,
            entry_typesig)
        return self

    def __call__(self, *args):
        return self._c_function(*args)


class MapTransformer(ast.NodeTransformer):
    def __init__(self, loopvar, param_dict, retval_name):
        self.loopvar = loopvar
        self.param_dict = param_dict
        self.retval_name = retval_name

    def visit_SymbolRef(self, node):
        if node.name in self.param_dict:
            return C.ArrayRef(node, C.SymbolRef(self.loopvar))
        return node

    def visit_Return(self, node):
        node.value = self.visit(node.value)
        return C.Assign(C.ArrayRef(C.SymbolRef(self.retval_name),
                                   C.SymbolRef(self.loopvar)),
                        node.value)


hwacha_configure_block = """
size_t vector_length;
__asm__ volatile (
    "vsetcfg 16, 1\\n"
    "vsetvl %0, %1\\n"
    : "=r"(vector_length)
    : "r"({SIZE})
);
"""

bounds_check = """
if ({SIZE} == {loopvar}) continue;
"""

class ScalarFinder(ast.NodeVisitor):
    def __init__(self, scalars):
        self.scalars = scalars

    def visit_Constant(self, node):
        self.scalars.add(node.value)

def get_scalars_in_body(node):
    scalars = set()
    visitor = ScalarFinder(scalars)
    for stmt in node.body:
        visitor.visit(stmt)
    return scalars

number_dict = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "0": "zero",
    ".": "dot"
}

def scalar_init(scalar):
    name = "".join(number_dict[digit] for digit in str(scalar))
    return StringTemplate("""
union {{
    float f;
    uint32_t i;
}} {name};
{name}.f = {scalar}f;
    """.format(name=name, scalar=scalar))

obtained_vector_length = """
size_t obtained_vector_length;
__asm__ volatile(
    "vsetvl %0, %1\\n"
    : "=r"(obtained_vector_length)
    : "r"({SIZE} - {loopvar})
    );
assert(obtained_vector_length <= {SIZE});
"""

class ArrayRefFinder(ast.NodeVisitor):
    def __init__(self, refs):
        self.refs = refs

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            self.refs.append(node)
        else:
            self.visit(node.left)
            self.visit(node.right)

def get_array_references_in_body(node):
    refs = []
    finder = ArrayRefFinder(refs)
    for stmt in node.body:
        finder.visit(stmt)
    return refs

class HwachaASMTranslator(ast.NodeTransformer):
    def __init__(self, scalars, ref_register_map, body, type_map):
        self.scalars = scalars
        self.ref_register_map = ref_register_map
        self.body = body
        self.curr_register = -1
        self.reg_map = {}
        self.type_map = type_map

    def get_next_register(self):
        self.curr_register += 1
        return "vv{}".format(self.curr_register)

    def visit_SymbolRef(self, node):
        if node.name in self.reg_map:
            return self.reg_map[node.name]
        return node

    def visit_Cast(self, node):
        reg = self.get_next_register()
        value = self.visit(node.value)
        if isinstance(node.type, ct.c_float):
            self.body.append("  vfcvt.s.w {0}, {1}\\n".format(reg, value))
            self.type_map[reg] = ct.c_float
            return reg
        else:
            raise NotImplementedError()

    def visit_Constant(self, node):
        self.type_map[node.value] = get_ctype(node.value)
        return self.scalars[node.value]

    def visit_FunctionCall(self, node):
        if node.func.name == 'max':
            arg1 = self.visit(node.args[0])
            arg2 = self.visit(node.args[1])
            reg = self.get_next_register()
            print(node)
            print(arg1)
            if self.type_map[arg1] == ct.c_float or \
                self.type_map[arg2] == ct.c_float:
                    self.body.append("  vfmax.s {0}, {1}, {2}\\n".format(
                        reg, arg1, arg2
                        ))
                    self.type_map[reg] = ct.c_float
                    return reg
        elif node.func.name == 'min':
            arg1 = self.visit(node.args[0])
            arg2 = self.visit(node.args[1])
            reg = self.get_next_register()
            if self.type_map[arg1] == ct.c_float or \
                self.type_map[arg2] == ct.c_float:
                    self.body.append("  vfmin.s {0}, {1}, {2}\\n".format(
                        reg, arg1, arg2
                        ))
                    self.type_map[reg] = ct.c_float
                    return reg
        raise NotImplementedError()

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            reg = self.get_next_register()
            self.body.append("  vlwu {0}, {1}\\n".format(
                    reg,
                    self.ref_register_map[str(node)][1]))
            return reg
        if isinstance(node.op, C.Op.Assign):
            node.right = self.visit(node.right)
            if isinstance(node.left, C.SymbolRef):
                self.reg_map[node.left.name] = node.right
                return
            elif isinstance(node.left, C.BinaryOp) and \
                    isinstance(node.left.op, C.Op.ArrayRef):
                if self.type_map[node.left.left.name] != self.type_map[node.right]:
                    reg = self.get_next_register()
                    self.body.append("  vfcvt.w.s {0}, {1}\\n".format(reg, node.right))
                    self.body.append("  vsw {0}, {1}\\n".format(reg,
                        self.ref_register_map[str(node.left)][1]))
                    return

        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        reg = self.get_next_register()
        if isinstance(node.op, C.Op.Sub):
            self.body.append("  vsub {0}, {1}, {2}\\n".format(
                reg, node.left, node.right))
        elif isinstance(node.op, C.Op.Div):
            if self.type_map[node.left] == ct.c_float or \
                self.type_map[node.right] == ct.c_float:
                    self.body.append("  vfdiv.s {0}, {1}, {2}\\n".format(
                        reg, node.left, node.right))
                    self.type_map[reg] = ct.c_float
            else:
                raise NotImplementedError()
        elif isinstance(node.op, C.Op.Mul):
            if self.type_map[node.left] == ct.c_float or \
                self.type_map[node.right] == ct.c_float:
                    self.body.append("  vfmul.s {0}, {1}, {2}\\n".format(
                        reg, node.left, node.right))
                    self.type_map[reg] = ct.c_float
            else:
                raise NotImplementedError()
        return reg

def get_asm_body(node, scalars, refs, type_map):
    body = """
__asm__ volatile (
".align 3\\n"
"__hwacha_body:\\n"
    """
    asm_body = []
    translator = HwachaASMTranslator(scalars, refs, asm_body, type_map)
    for s in node.body:
        translator.visit(s)
    for s in asm_body:
        body += "\"" + s + "\"\n"
    body += "\"  vstop\\n\"\n"
    body += "  );"
    return StringTemplate(body)


class HwachaVectorize(ast.NodeTransformer):
    def __init__(self, type_map, defns):
        self.type_map = type_map
        self.defns = defns

    def visit_For(self, node):
        if node.pragma == "ivdep":
            block = []
            loopvar = node.incr.arg
            size = node.test.right
            scalars = get_scalars_in_body(node)
            refs = get_array_references_in_body(node)
            ref_register_map = {}
            scalar_register_map = {}
            for index, ref in enumerate(refs):
                ref_register_map[str(ref)] = (ref, "va{}".format(index))
            for index, scalar in enumerate(scalars):
                reg = "vs{}".format(index)
                scalar_register_map[scalar] = reg
                self.type_map[reg] = get_ctype(scalar)
            body = []
            block.append(StringTemplate(hwacha_configure_block.format(SIZE=size)))

            node.incr = C.AddAssign(loopvar, C.SymbolRef("vector_length"))
            self.defns.append(get_asm_body(node, scalar_register_map,
                ref_register_map, self.type_map))
            block.append(node)

            body.append(StringTemplate(bounds_check.format(SIZE=size,
                loopvar=loopvar)))

            for scalar in scalars:
                body.append(scalar_init(scalar))

            body.append(StringTemplate(obtained_vector_length.format(SIZE=size,
                loopvar=loopvar)))

            block1 = ""
            block2 = ""
            index = 0
            for _, info in ref_register_map.items():
                ref, register = info
                block1 += "\t \"vmsa {0}, %{1}\\n\"\n".format(register, index)
                block2 += "\"r\"({0} + {1}),\n".format(
                    ref.left.name, ref.right.name)
                index += 1
            for scalar, register in scalar_register_map.items():
                block1 += "\t \"vmss {0}, %{1}\\n\"\n".format(register, index)
                block2 += "\"r\"({0}.i),\n".format(
                    "".join(number_dict[digit] for digit in str(scalar)))
                index += 1
            block1 += "\"fence\\n\"\n"
            block1 += "\"vf 0(%{0})\\n\"\n".format(index)
            block2 += "\"r\" (&__hwacha_body)"
            body.append(StringTemplate(
                """
__asm__ volatile(
{block1}
    :
    : {block2}
    : "memory"
);
                """.format(block1=block1, block2=block2)))

            node.body = body
            block.append(
StringTemplate("""
__asm__ volatile(
    "fence\\n"
);
"""))
            return block



class HwachaTranslator(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return tuple(get_nd_pointer(arg) for arg in args)

    def transform(self, py_ast, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        tree = PyBasicConversions().visit(py_ast)
        param_dict = {}
        tree.body[0].params.append(C.SymbolRef("retval", arg_cfg[0]()))
        # Annotate arguments
        for param, type in zip(tree.body[0].params, arg_cfg):
            param.type = type()
            param_dict[param.name] = type._dtype_

        length = np.prod(arg_cfg[0]._shape_)
        transformer = MapTransformer("i", param_dict, "retval")
        body = list(map(transformer.visit, tree.body[0].defn))

        tree.body[0].defn = [C.For(
                C.Assign(C.SymbolRef("i", ct.c_int()), C.Constant(0)),
                C.Lt(C.SymbolRef("i"), C.Constant(length)),
                C.PostInc(C.SymbolRef("i")),
                body=body,
                pragma="ivdep"
            )]

        tree = DeclarationFiller().visit(tree)
        defns = []
        tree = HwachaVectorize(param_dict, defns).visit(tree)
        file_body = [
            StringTemplate("#include <stdlib.h>"),
            StringTemplate("#include <stdint.h>"),
            StringTemplate("#include <assert.h>"),
            StringTemplate("extern \"C\" void __hwacha_body(void);"),
        ]
        file_body.extend(defns)
        file_body.append(tree)
        return [CFile("generated", file_body)]

    def finalize(self, transform_result, program_config):
        generated = transform_result[0]
        print(generated)
        proj = Project([generated])
        entry_type = ct.CFUNCTYPE(None, *program_config[0])
        return HwachaFN().finalize("apply", proj, entry_type)


def hwacha_map(fn, *args):
    mapfn = HwachaTranslator.from_function(fn, "map")
    retval = np.empty_like(args[0])
    args += (retval, )
    mapfn(*args)
    return retval


CALIBRATE_COLD = 0x7000
CALIBRATE_HOT  = 0xA000

SIZE = (208 * 156)

# Generate a dummy calibration table, just so there's something
# to execute.
cold = np.full(SIZE, CALIBRATE_COLD, np.int32)
hot  = np.full(SIZE, CALIBRATE_HOT, np.int32)

# Generate a dummy input image, again just so there's something
# to execute.
raw  = np.empty(SIZE, np.int32)

for i in range(SIZE):
    scale = (CALIBRATE_HOT - CALIBRATE_COLD)
    percent = (i % 120) - 10
    raw[i] = scale * (percent / 100.0) + CALIBRATE_COLD
    raw[i] = CALIBRATE_COLD + (i % (int)(scale - 2)) + 1

def gold(cold, hot, raw, flat):
    for i in range(208 * 156):
        _max = hot[i]
        _min = cold[i]
        offset = raw[i] - _min
        scale = _max - _min
        foffset = float(offset)
        fscale = float(scale)
        scaled = foffset / fscale
        scaled = min(1.0, scaled)
        scaled = max(0.0, scaled)
        flat[i] = 255 * scaled

def test_map(cold, hot, raw):
    _max = hot
    _min = cold
    offset = raw - _min
    scale = _max - _min
    foffset = float(offset)
    fscale = float(scale)
    scaled = foffset / fscale
    scaled = min(1.0, scaled)
    scaled = max(0.0, scaled)
    return 255.0 * scaled


flat_gold = np.empty_like(raw)
gold(cold, hot, raw, flat_gold)

flat_test = hwacha_map(test_map, cold, hot, raw)

np.testing.assert_array_equal(flat_gold, flat_test)
