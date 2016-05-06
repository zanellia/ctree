from ctree.simd.types import m256d, m256, m512

from ctree.types import register_type_codegenerators

register_type_codegenerators({
    m256d: lambda t: "__m256d",
    m256: lambda t: "__m256",
    m512: lambda t: "__m512"
})
