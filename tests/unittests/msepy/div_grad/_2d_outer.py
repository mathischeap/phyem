# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 12:52 PM on 5/24/2023

$ python tests/unittests/msepy/div_grad/_2d_outer.py
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

import numpy as np

import __init__ as ph
n = 2
ls, mp = ph.samples.wf_div_grad(n=n, degree=8, orientation='outer', periodic=False)
# print(ls._bc)
# ls.pr()

# msepy, obj = ph.fem.apply('msepy', locals())
#
# manifold = msepy.base['manifolds'][r"\mathcal{M}"]
# boundary_manifold = msepy.base['manifolds'][r"\partial\mathcal{M}"]
# Gamma_phi = msepy.base['manifolds'][r"\Gamma_\phi"]
# Gamma_u = msepy.base['manifolds'][r"\Gamma_u"]
#
# # msepy.config(manifold)(
# #     'crazy_multi.rst', c=0.1, bounds=[[0, 1] for _ in range(n)], periodic=False,
# # )
# msepy.config(manifold)('backward_step')
# msepy.config(Gamma_phi)(manifold, {0: [1, 0, 0, 0]})
#
# # manifold.visualize()
# # boundary_manifold.visualize()
# # Gamma_phi.visualize()
# # Gamma_u.visualize()
#
# mesh = msepy.base['meshes'][r'\mathfrak{M}']
# msepy.config(mesh)([5, 5])
#
# #
# phi = msepy.base['forms']['potential']
# u = msepy.base['forms']['velocity']
# f = msepy.base['forms']['source']
#
# # ls = obj['ls']
# # ls.pr()
#
# ls = obj['ls'].apply()
# ls.pr()
#
#
# def phi_func(t, x, y):
#     """"""
#     return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0


# phi_scalar = ph.vc.scalar(phi_func)
# phi.cf = phi_scalar
# u.cf = phi.cf.codifferential()
# f.cf = - u.cf.exterior_derivative()
# f[0].reduce()
# phi[0].reduce()
# f[0].visualize()

# ls0 = ls(0)
# ls0.customize.set_dof(-1, phi[0].cochain.of_dof(-1))
# als = ls0.assemble()
# als.solve()
#
# phi[0].visualize()
# # u[0].visualize()
# print(phi[0].error())
# print(u[0].error())
