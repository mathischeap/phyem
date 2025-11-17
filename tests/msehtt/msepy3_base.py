# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt mesh built upon msepy 3d meshes.

mpiexec -n 4 python tests/msehtt/msepy3_base.py
"""
import numpy as np
import phyem as ph

ph.config.set_embedding_space_dim(3)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)


N = 3
K = 8
c = 0.

manifold = ph.manifold(3, periodic=False)
mesh = ph.mesh(manifold)

Out0 = ph.space.new('Lambda', 0, orientation='outer')
Out1 = ph.space.new('Lambda', 1, orientation='outer')
Out2 = ph.space.new('Lambda', 2, orientation='outer')
Out3 = ph.space.new('Lambda', 3, orientation='outer')

Inn0 = ph.space.new('Lambda', 0, orientation='inner')
Inn1 = ph.space.new('Lambda', 1, orientation='inner')
Inn2 = ph.space.new('Lambda', 2, orientation='inner')
Inn3 = ph.space.new('Lambda', 3, orientation='inner')

o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')
o3 = Out3.make_form(r'\tilde{\omega}^3', 'outer-form-3')

i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')
i3 = Inn3.make_form(r'{\omega}^3', 'inner-form-3')

ph.space.finite(N)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=c, periodic=False)
_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')

# _mesh.visualize()


of0 = obj['o0']
of1 = obj['o1']
of2 = obj['o2']
of3 = obj['o3']

if0 = obj['i0']
if1 = obj['i1']
if2 = obj['i2']
if3 = obj['i3']


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

of0[0].reduce()
of1[0].reduce()
of2[0].reduce()
of3[0].reduce()

from scipy.integrate import nquad
NORM_f = nquad(f_norm, ([0, 1], [0, 1], [0, 1]))[0] ** 0.5
NORM_w = nquad(w_norm, ([0, 1], [0, 1], [0, 1]))[0] ** 0.5

norm = of0[0].norm()
np.testing.assert_almost_equal(norm, NORM_w)
norm = of1[0].norm()
np.testing.assert_almost_equal(norm, NORM_f, decimal=5)
norm = of2[0].norm()
np.testing.assert_almost_equal(norm, NORM_f, decimal=5)
norm = of3[0].norm()
np.testing.assert_almost_equal(norm, NORM_w, decimal=5)

of0.cf = scalar
of0[0].reduce()
E = of0.incidence_matrix
of1[0].cochain = E @ of0[0].cochain
of1.cf = of0.cf.exterior_derivative()
error = of1[0].error()
assert error < 0.006

of1.cf = vector
of1[0].reduce()
E = of1.incidence_matrix
of2[0].cochain = E @ of1[0].cochain
of2.cf = of1.cf.exterior_derivative()
error = of2[0].error()
assert error < 0.006

of2.cf = vector
of2[0].reduce()
E = of2.incidence_matrix
of3[0].cochain = E @ of2[0].cochain
of3.cf = of2.cf.exterior_derivative()
error = of3[0].error()
assert error < 0.02
