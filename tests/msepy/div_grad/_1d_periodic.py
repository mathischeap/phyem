# -*- coding: utf-8 -*-
r"""
"""

# python tests/msepy/div_grad/_1d_periodic.py
import sys

if './' not in sys.path:
    sys.path.append('./')
import numpy as np
import __init__ as ph

ls = ph.samples.wf_div_grad(n=1, degree=4, orientation='outer', periodic=True)
# ls.pr()

msepy, obj = ph.fem.apply('msepy', locals())

manifold = msepy.base['manifolds'][r"\mathcal{M}"]
mesh = msepy.base['meshes'][r'\mathfrak{M}']

msepy.config(manifold)(
    'crazy_multi', c=0.3, bounds=[[0, 1], ], periodic=True,
)
msepy.config(mesh)(20)

phi = msepy.base['forms'][r'potential']
u = msepy.base['forms'][r'velocity']
f = msepy.base['forms'][r'source']

# ls.pr()

ls = obj['ls'].apply()

# ls.pr()


def phi_func(t, x):
    """"""
    return np.sin(2 * np.pi * x) + t * 0


phi_scalar = ph.vc.scalar(phi_func)
phi.cf = phi_scalar
u.cf = phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[0].reduce()
phi[0].reduce()

ls0 = ls(0)

ls0.customize.set_dof(-1, phi[0].cochain.of_dof(-1))
als = ls0.assemble()
als.solve()
# print(als.solve._last_solver_message)

# phi[0].visualize()
# u[0].visualize()
print(phi[0].error())
print(u[0].error())
