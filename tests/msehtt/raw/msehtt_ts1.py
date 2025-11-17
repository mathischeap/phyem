# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/raw/msehtt_ts1.py
"""

import sys

import numpy as np

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 4

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


# ------- manually make a vtu interface file -------------------------------------


from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface

from random import uniform, randint

from src.config import MASTER_RANK, RANK, COMM


if RANK == MASTER_RANK:
    # __ = [uniform(-0.1, 0.1) for _ in range(8)]
    __ = [0 for _ in range(8)]

else:
    __ = None

a, b, c, d, e, f, g, h = COMM.bcast(__, root=MASTER_RANK)

coo = {
    0: (0, 0),
    1: (0, 0.5),
    2: (0, 1),
    3: (0.5, 1),
    4: (1, 1),
    5: (1, 0.5),
    6: (1, 0),
    7: (0.5, 0),
    8: (0.5, 0.5),
    9: (0.25 + a, 0.25 + b),
    10: (0.25 + c, 0.75 + d),
    11: (0.75 + e, 0.75 + f),
    12: (0.75 + g, 0.25 + h),
    13: (1.5, 0),
    14: (1.5, 0.5),
    15: (1.5, 1),
    16: (2, 0),
    17: (2, 0.5),
    18: (2, 1),
}

connections = {
    '0t0': [0, 9, 1],
    '0t1': [1, 9, 10],
    (2, 'S'): [2, 1, 10],
    '3': [2, 10, 3],
    't4': [10, 8, 11],
    '5': [3, 10, 11],
    6: [11, 5, 4],
    7: [3, 11, 4],
    8: [11, 12, 5],
    '9': [8, 12, 11],
    10: [8, 9, 12],
    11: [10, 9, 8],
    12: [9, 7, 12],
    't13': [7, 6, 12],
    14: [12, 6, 5],
    15: [9, 0, 7],
    'q0': [5, 6, 13, 14],
    'q1': [5, 14, 15, 4],
    'q3': [13, 16, 17, 14],
    'q4': [15, 14, 17, 18],
}


cell_types = {}
for e in connections:
    cell_types[e] = 5
cell_types['q0'] = 9
cell_types['q1'] = 9
cell_types['q3'] = 9
cell_types['q4'] = 9


if RANK == MASTER_RANK:
    __ = [randint(0, 3) for _ in range(20)]
else:
    __ = None
__ = COMM.bcast(__, root=MASTER_RANK)

CONNECTIONS = {}

for i, e in enumerate(connections):
    nodes = connections[e]
    rolling = __[i]
    if e in ['q0', 'q1', 'q3', 'q4']:
        if rolling == 0:
            pass
        elif rolling == 1:
            nodes = [nodes[1], nodes[2], nodes[3], nodes[0]]
        elif rolling == 2:
            nodes = [nodes[2], nodes[3], nodes[0], nodes[1]]
        elif rolling == 3:
            nodes = [nodes[3], nodes[0], nodes[1], nodes[2]]
        else:
            raise Exception()
    else:
        if rolling == 0:
            pass
        elif rolling == 1:
            nodes = [nodes[1], nodes[2], nodes[0]]
        elif rolling == 2:
            nodes = [nodes[2], nodes[0], nodes[1]]
        else:
            pass

    CONNECTIONS[e] = nodes

vif = MseHttVtuInterface(coo, CONNECTIONS, cell_types)

ph.space.finite(N)

# -----------------------------------------------------------------------------------

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(vif, ts=1)
tgm.visualize(quality=False, internal_grid=0)

# msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
# msehtt.config(msehtt_mesh)(tgm, including='all')
# # msehtt_mesh.visualize()
#
# fi0 = obj['i0']
# fi1 = obj['i1']
# fi2 = obj['i2']
#
# fo0 = obj['o0']
# fo1 = obj['o1']
# fo2 = obj['o2']
#
#
# def fx(t, x, y):
#     return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)
#
#
# def fy(t, x, y):
#     return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)
#
#
# def fw(t, x, y):
#     return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(t)
#
#
# vector = ph.vc.vector(fx, fy)
# scalar = ph.vc.scalar(fw)
#
# fo0.cf = scalar
# fo0[0].reduce()
# fo1.cf = vector
# fo1[0].reduce()
# fo2.cf = scalar
# fo2[0].reduce()
#
# # fo0[0].visualize.matplot()
# # fo1[0].visualize()
# # fo2[0].visualize.matplot()
#
# err0 = fo0[0].error()
# err1 = fo1[0].error()
# err2 = fo2[0].error()
# print(err0, err1, err2)

# fi0.cf = scalar
# fi0[0].reduce()
# fi1.cf = vector
# fi1[0].reduce()
# fi2.cf = scalar
# fi2[0].reduce()
#
# # fi0[0].visualize()
# # fi1[0].visualize()
# fi2[0].visualize.matplot()
#
# err0 = fi0[0].error()
# err1 = fi1[0].error()
# err2 = fi2[0].error()
# print(err0, err1, err2)
#
# # fo0[1].reduce()
# # E = fo0.incidence_matrix
# # fo1[1].cochain = E @ fo0[1].cochain
# # fo1.cf = fo0.cf.exterior_derivative()
# # error = fo1[1].error()
# # assert error < 1e-3
# #
# # fo1[1].reduce()
# # E = fo1.incidence_matrix
# # fo2[1].cochain = E @ fo1[1].cochain
# # fo2.cf = fo1.cf.exterior_derivative()
# # error = fo2[1].error()
# # assert error < 1e-3
# #
# # fi0[1].reduce()
# # E = fi0.incidence_matrix
# # fi1[1].cochain = E @ fi0[1].cochain
# # fi1.cf = fi0.cf.exterior_derivative()
# # error = fi1[1].error()
# # assert error < 1e-3
# #
# # fi1[1].reduce()
# # E = fi1.incidence_matrix
# # fi2[1].cochain = E @ fi1[1].cochain
# # fi2.cf = fi1.cf.exterior_derivative()
# # error = fi2[1].error()
# # assert error < 1e-3
