# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt_vtu/mesh0.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 5

manifold = ph.manifold(2, periodic=False)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

Out1 = ph.space.new('Lambda', 1, orientation='outer')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o2 = Out2.make_form(r'\tilde{f}^2', 'outer-form-2')

i0 = Inn0.make_form(r'\omega^0', 'inner-form-0')
i2 = Inn2.make_form(r'f^2', 'inner-form-2')

i1 = Inn1.make_form(r'u^1', 'inner-form-1')
o1 = Out1.make_form(r'\tilde{u}^1', 'outer-form-1')

# ------- manually make a vtu interface file ---------------
from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface

from random import uniform

shift1 = uniform(-0.4, 0.4)
shift2 = uniform(-0.4, 0.4)
# shift1 = 0
# shift2 = 0

coo = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
    4: (0.5 + shift1, 0.5 + shift2),
    5: (1.5, shift2),
    6: (1.5, 1 + shift1),
    7: (2, shift2),
    8: (2, 1 + shift1),
}

connections = {
    # '0t0': [4, 1, 0],
    # '0t1': [4, 2, 1],
    # '0t2': [4, 3, 2],
    # '0t3': [4, 0, 3],

    '0t0': [4, 1, 0],
    '0t1': [2, 1, 4],
    '0t2': [2, 4, 3],
    '0t3': [3, 4, 0],

    # 4: [2, 3, 5, 6],
    # 5: [5, 7, 6, 8],
}

cell_types = {
    '0t0': 5,
    '0t1': 5,
    '0t2': 5,
    '0t3': 5,
    # 4: 9,
    # 5: 8,
}

vif = MseHttVtuInterface(coo, connections, cell_types)

ph.space.finite(N)

# -------------------------------------------------------------

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(vif)

# tgm.visualize()

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

fi0 = obj['i0']
fi1 = obj['i1']
fi2 = obj['i2']

fo0 = obj['o0']
fo1 = obj['o1']
fo2 = obj['o2']


def fx(t, x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)


def fy(t, x, y):
    return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)


def fw(t, x, y):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)


def f_norm(x, y):
    return fx(0, x, y)**2 + fy(0, x, y)**2


def fw0_square(x, y):
    return fw(0, x, y)**2


vector = ph.vc.vector(fx, fy)
scalar = ph.vc.scalar(fw)

fo0.cf = scalar
fo0[0].reduce()
err0 = fo0[0].error()
# print(err0)

fo2.cf = scalar
fo2[0].reduce()
# fo2[0].visualize()
err2 = fo2[0].error()
# print(err2)

fo1.cf = vector
fo1[0].reduce()
# fo1[0].visualize()
err1o = fo1[0].error()
# print(err1o)

fi1.cf = vector
fi1[0].reduce()
# fo1[0].visualize()
err1i = fi1[0].error()
# print(err1o)

# E = fo0.incidence_matrix
# fo1[0].cochain = E @ fo0[0].cochain
# # fo1[0].visualize()
#
# fi0.cf = scalar
# fi0[0].reduce()
# E = fi0.incidence_matrix
# fi1[0].cochain = E @ fi0[0].cochain

# E = fi1.incidence_matrix
# fi2[0].cochain = E @ fi1[0].cochain
# fi2[0].visualize()

E = fo1.incidence_matrix
fo2[0].cochain = E @ fo1[0].cochain
fo2[0].visualize()

# fo1[0].visualize()
#
#
# fi1.cf = vector
# fi1[0].reduce()
# # fi1[0].visualize()
# err1i = fi1[0].error()
# # print(err1i)
