# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 11:53 AM on 5/17/2023

$ python tests/unittests/msepy/div_grad/_3d_outer_periodic.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
import numpy as np
import __init__ as ph

ls = ph.samples.wf_div_grad(n=3, degree=2, orientation='outer', periodic=True)
# ls.pr()

msepy, obj = ph.fem.apply('msepy', locals())

manifold = msepy.base['manifolds'][r"\mathcal{M}"]
mesh = msepy.base['meshes'][r'\mathfrak{M}']

msepy.config(manifold)(
    'crazy', c=0., bounds=[[0, 1], [0, 1], [0, 1]], periodic=True,
)
msepy.config(mesh)([6, 6, 6])

phi = msepy.base['forms'][r'potential']
u = msepy.base['forms'][r'velocity']
f = msepy.base['forms'][r'source']

# ls.pr()
ls = obj['ls'].apply()


def phi_func(t, x, y, z):
    """"""
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 0


phi_scalar = ph.vc.scalar(phi_func)
phi.cf = phi_scalar
u.cf = phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[0].reduce()
# phi[0].reduce()

ls0 = ls(0)
ls0.customize.set_dof(-1, 0)
als = ls0.assemble()
# als.solve()
#
# print(phi[0].error())
# print(u[0].error())
