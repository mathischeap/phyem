# -*- coding: utf-8 -*-
r"""
python tests/msepy/div_grad/_2d_inner.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

import numpy as np

import __init__ as ph
n = 2
ls, mp = ph.samples.wf_div_grad(n=n, degree=3, orientation='inner', periodic=False)
# ls.pr()

msepy, obj = ph.fem.apply('msepy', locals())

manifold = msepy.base['manifolds'][r"\mathcal{M}"]
boundary_manifold = msepy.base['manifolds'][r"\partial\mathcal{M}"]
Gamma_phi = msepy.base['manifolds'][r"\Gamma_\phi"]
Gamma_u = msepy.base['manifolds'][r"\Gamma_u"]

msepy.config(manifold)(
    'crazy', c=0., bounds=[[0., 1.] for _ in range(n)], periodic=False,
)
# msepy.config(manifold)('backward_step')
msepy.config(Gamma_u)(
    manifold, {0: [1, 1, 0, 0]}
)

mesh = msepy.base['meshes'][r'\mathfrak{M}']
msepy.config(mesh)([12, 12])

# for mesh_repr in msepy.base['meshes']:
#     mesh = msepy.base['meshes'][mesh_repr]
#     mesh.visualize()

phi = msepy.base['forms']['potential']
u = msepy.base['forms']['velocity']
f = msepy.base['forms']['source']

ls = obj['ls'].apply()


def phi_func(t, x, y):
    """"""
    return - np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + t * 0


phi_scalar = ph.vc.scalar(phi_func)
phi.cf = phi_scalar
u.cf = phi_scalar.gradient
f.cf = - phi_scalar.gradient.divergence

ls.bc.config(Gamma_u)(phi_scalar.gradient)   # natural
ls.bc.config(Gamma_phi)(phi.cf)   # essential

f[0].reduce()

ls0 = ls(0)

als = ls0.assemble()
als.solve()

# phi[0].visualize()
# u[0].visualize()

print(phi[0].error())
print(u[0].error())
