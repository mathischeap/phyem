# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt mesh built upon msepy 2d meshes.

mpiexec -n 4 python tests/msehtt_vtu/base2_refining.py
"""

import numpy as np
import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 3
K = 12
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

boundary = mesh.boundary()

ph.space.finite(N)


# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=c, periodic=False, trf=1)
# tgm.visualize()

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)
total_boundary.visualize()

fi0 = obj['i0']
fi1 = obj['i1']
fi2 = obj['i2']

fo0 = obj['o0']
fo1 = obj['o1']
fo2 = obj['o2']


def fx(t, x, y):
    return np.sin(2 * np.pi * x) * np.cos(1.24 * np.pi * y) * np.exp(t)


def fy(t, x, y):
    return np.cos(1.77 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(t)


def fw(t, x, y):
    return np.cos(2 * np.pi * x) * np.cos(1.99 * np.pi * y) * np.exp(t)


vector = ph.vc.vector(fx, fy)
scalar = ph.vc.scalar(fw)

fo0.cf = scalar
fo0[0].reduce()
fo1.cf = vector
fo1[0].reduce()
fo2.cf = scalar
fo2[0].reduce()
err0 = fo0[0].error()
err1 = fo1[0].error()
err2 = fo2[0].error()
print(err2)

fi0.cf = scalar
fi0[0].reduce()
fi1.cf = vector
fi1[0].reduce()
fi2.cf = scalar
fi2[0].reduce()
err0 = fi0[0].error()
err1 = fi1[0].error()
err2 = fi2[0].error()
# print(err0, err1, err2)

fo0[1].reduce()
E = fo0.incidence_matrix
fo1[1].cochain = E @ fo0[1].cochain
fo1.cf = fo0.cf.exterior_derivative()
error = fo1[1].error()
print(error)
# assert error < 1e-3

fo1[1].reduce()
E = fo1.incidence_matrix
fo2[1].cochain = E @ fo1[1].cochain
fo2.cf = fo1.cf.exterior_derivative()
error = fo2[1].error()
print(error)
# assert error < 1e-3

fi0[1].reduce()
E = fi0.incidence_matrix
fi1[1].cochain = E @ fi0[1].cochain
fi1.cf = fi0.cf.exterior_derivative()
error = fi1[1].error()
print(error)
# assert error < 1e-3

fi1[1].reduce()
E = fi1.incidence_matrix
fi2[1].cochain = E @ fi1[1].cochain
fi2.cf = fi1.cf.exterior_derivative()
error = fi2[1].error()
print(error)
# assert error < 1e-3
