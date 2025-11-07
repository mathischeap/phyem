# -*- coding: utf-8 -*-
"""
Test the time-space function of forms (through form.export)

mpiexec -n 4 python tests/msehtt/tsf3_save_read_reduce_tests.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph

ph.config.set_embedding_space_dim(3)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)


N = 2
K = 4
c = 0.

manifold = ph.manifold(3, periodic=False)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')
Out3 = ph.space.new('Lambda', 3, orientation='outer')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')
o3 = Out3.make_form(r'\tilde{\omega}^3', 'outer-form-3')

O0 = Out0.make_form(r'\tilde{\Omega}^0', 'Outer-form-0')
O1 = Out1.make_form(r'\tilde{\Omega}^1', 'Outer-form-1')
O2 = Out2.make_form(r'\tilde{\Omega}^2', 'Outer-form-2')
O3 = Out3.make_form(r'\tilde{\Omega}^3', 'Outer-form-3')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=c, periodic=False)
_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')


of0 = obj['o0']
of1 = obj['o1']
of2 = obj['o2']
of3 = obj['o3']

Of0 = obj['O0']
Of1 = obj['O1']
Of2 = obj['O2']
Of3 = obj['O3']


def fx(t, x, y, z):
    return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) * np.exp(t)


def fy(t, x, y, z):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z) * np.exp(t)


def fz(t, x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(1.5 * np.pi * z) * np.exp(t)


def fw(t, x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) * np.exp(t)


def f_norm(x, y, z):
    return fx(0, x, y, z)**2 + fy(0, x, y, z)**2 + fz(0, x, y, z)**2


def w_norm(x, y, z):
    return fw(0, x, y, z)**2


vector = ph.vc.vector(fx, fy, fz)
scalar = ph.vc.scalar(fw)

of0.cf = scalar
of1.cf = vector
of2.cf = vector
of3.cf = scalar

Of0.cf = scalar
Of1.cf = vector
Of2.cf = vector
Of3.cf = scalar

of0[0].reduce()
of1[0].reduce()
of2[0].reduce()
of3[0].reduce()

# of0[None].export.tsf('of0_tsf')
# tsf = ph.read_tsf('of0_tsf')
# Of0[0].reduce(tsf[0])
# error = of0[0].error()
# Error = Of0[0].error()
# print(error, Error)
# ph.os.remove('of0_tsf')

of3[None].export.tsf('of3_tsf')
tsf = ph.read_tsf('of3_tsf')
Of3[0].reduce(tsf[0])
error = of3[0].error()
Error = Of3[0].error()
assert abs(error - Error) < 0.005
ph.os.remove('of3_tsf')
