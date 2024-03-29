#!/usr/bin/env python
# coding=utf-8

import pdb
from sympy import Symbol, simplify
from sympy.matrices import Matrix

i = Symbol("i")
d0 = Symbol("d0")
d1 = Symbol("d1")
d2 = Symbol("d2")

vd = Matrix([d0, d1, d2])
m = Matrix([[i**2, i, 1], [(i - 1) ** 2, i - 1, 1], [(i - 2) ** 2, i - 2, 1]])
v2 = Matrix([(i + 1) ** 2, i + 1, 1])

d2 = v2.T * m.inv() * vd
print(simplify(d2))
