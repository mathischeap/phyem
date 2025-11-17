# -*- coding: utf-8 -*-
"""
mpiexec -n 4 python tests/msehtt/Poisson3.py
"""

import numpy as np

import phyem as ph
ph.config.set_pr_cache(False)

N = 3
K = 5
c = 0.

time = 1

ls = ph.samples.wf_div_grad(n=3, degree=N, orientation='outer', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=c,
    bounds=([0.25, 1.25], [0.25, 1.25], [0.25, 1.25]),
    periodic=False
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)
# total_boundary.visualize()

boundary_u = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_u\right)"]
boundary_p = msehtt.base['meshes'][r"\partial\mathfrak{M}\left(\Gamma_\phi\right)"]

msehtt.config(boundary_p)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([-1, 0, 0], [0, -1, 0], [0, 0, 1])    # outward unit norm vector.
    }
)
# boundary_p.visualize()

msehtt.config(boundary_u)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'except ounv': ([-1, 0, 0], [0, -1, 0], [0, 0, 1])    # outward unit norm vector.
    }
)
# boundary_u.visualize()

phi = msehtt.base['forms'][r'potential']
u = msehtt.base['forms'][r'velocity']
f = msehtt.base['forms'][r'source']

M3 = msehtt.array('mass matrix', phi)[0]  # manually make an array.
invM3 = M3.inv()


def phi_func(t, x, y, z):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t


phi_scalar = ph.vc.scalar(phi_func)

phi.cf = phi_scalar
# phi[time].reduce()
# phi[0].visualize.quick()

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[time].reduce()
# # f[0].visualize.quick()
#
msehtt_ls = obj['ls'].apply()
# msehtt_ls.pr()
# # print(msehtt_ls)
# # print(u.cf.field)
#
msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc

# print(msehtt_ls.customize)

linear_system = msehtt_ls(time)
# linear_system.spy(0)
# linear_system.customize.left_matmul_A_block(1, 0, invM3)
# linear_system.customize.left_matmul_b_block(1, invM3)
# linear_system.spy(0)
Axb = linear_system.assemble(threshold=1e-8)
# print(Axb.A.rank_nnz)
x, message, info = Axb.solve('spsolve')
linear_system.x.update(x)
# print(u[time].error(), phi[time].error())
assert phi[time].error() < 0.004
assert u[time].error() < 0.04

# print(u[time].error(), u[time].error('H1'))

# phi.numeric.export.scatter('phi', ddf=2)
#
# from src.config import RANK, MASTER_RANK
#
# if RANK == MASTER_RANK:
#     from scipy.interpolate import NearestNDInterpolator
#     import pickle
#     with open('phi', 'rb') as inputs:
#         objs = pickle.load(inputs)
#     inputs.close()
#
#     XYZ = objs['coordinates']
#     X, Y, Z = XYZ
#     V = objs['values'][0]
#     interp = NearestNDInterpolator(
#         list(zip(X, Y, Z)), V
#     )
#     x = np.linspace(0.25, 1.25, 200)
#     y = np.linspace(0.25, 1.25, 100)
#     z = np.array([0.25])
#
#     x, y, z = np.meshgrid(x, y, z, indexing='ij')
#
#     value = interp(x, y, z)
#
#     ph.tools.contour(x[:,:, 0], y[:,:, 0], value[:,:, 0])
