# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt-adaptive-mesh.

mpiexec -n 4 python tests/msehtt/adaptive/base2.py
"""


import numpy as np

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 5
K = 5
c = 0.

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')

i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-a', locals())
# print(obj)
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'chaotic', element_layout=K, c=c
    # 'meshpy',
    # points=(
    #     [0, 0],
    #     [0, -1],
    #     [1, -1],
    #     [1, 1],
    #     [-1, 1],
    #     [-1, 0],
    # ),
    # max_volume=0.2,
    # ts=1,
)
# tgm.visualize()

_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')

msehtt.initialize()

# cb = msehtt.current_base()

# _mesh.visualize()

fo0 = obj['o0']
fo1 = obj['o1']
fo2 = obj['o2']

fi0 = obj['i0']
fi1 = obj['i1']
fi2 = obj['i2']
#
# # print(fo0, fi0)


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
fi0.cf = scalar
fo1.cf = vector
fi1.cf = vector
fo2.cf = scalar
fi2.cf = scalar
#
fo0[0].reduce()
fi0[0].reduce()
fo1[0].reduce()
fi1[0].reduce()
fo2[0].reduce()
fi2[0].reduce()

assert fo0[None].error() < 1e-5
assert fi0[None].error() < 1e-5
assert fo1[None].error() < 1e-5
assert fi1[None].error() < 1e-5
assert fo2[None].error() < 1e-5
assert fi2[None].error() < 1e-5

from scipy.integrate import nquad
NORM = nquad(f_norm, ([0, 1], [0, 1]))[0] ** 0.5

norm = fo1[0].norm()
np.testing.assert_almost_equal(norm, NORM)
norm = fi1[0].norm()
np.testing.assert_almost_equal(norm, NORM)

NORM = nquad(fw0_square, ([0, 1], [0, 1]))[0] ** 0.5
norm = fo0[0].norm()
np.testing.assert_almost_equal(norm, NORM)
norm = fi0[0].norm()
np.testing.assert_almost_equal(norm, NORM)
norm = fo2[0].norm()
np.testing.assert_almost_equal(norm, NORM)
norm = fi2[0].norm()
np.testing.assert_almost_equal(norm, NORM)

fi0.cf = scalar
fi0[0].reduce()
E = fi0.incidence_matrix
fi1[0].cochain = E @ fi0[0].cochain
fi1.cf = fi0.cf.exterior_derivative()
error = fi1[0].error()
assert error < 1e-4

fi1.cf = vector
fi1[0].reduce()
E = fi1.incidence_matrix
fi2[0].cochain = E @ fi1[0].cochain
fi2.cf = fi1.cf.exterior_derivative()
error = fi2[0].error()
assert error < 1e-11

fo0.cf = scalar
fo0[0].reduce()
E = fo0.incidence_matrix
fo1[0].cochain = E @ fo0[0].cochain
fo1.cf = fo0.cf.exterior_derivative()
error = fo1[0].error()
assert error < 1e-4

fo1.cf = vector
fo1[0].reduce()
E = fo1.incidence_matrix
fo2[0].cochain = E @ fo1[0].cochain
fo2.cf = fo1.cf.exterior_derivative()
error = fo2[0].error()
assert error < 1e-4

df1 = fo1.d()
df1.cf = fo1.cf.exterior_derivative()
error = df1[None].error()
assert error < 2e-5

sub = fo2 - df1
error = sub[None].norm()
assert error == 0
