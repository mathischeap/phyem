# -*- coding: utf-8 -*-
"""
mpiexec -n 4 python tests/msehtt/Poisson3_inner.py
"""
import numpy as np

import phyem as ph
ph.config.set_pr_cache(False)

N = 3
K = 5
c = 0.

time = 1

ls = ph.samples.wf_div_grad(n=3, degree=N, orientation='inner', periodic=False)[0]
# ls.pr()
# print(ls._ls.A(0,0)[0][0]._lin_repr)

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


def phi_func(t, x, y, z):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t


phi_scalar = ph.vc.scalar(phi_func)

phi.cf = phi_scalar
# phi[time].reduce()
# phi[0].visualize.quick()
#
u.cf = phi.cf.exterior_derivative()
f.cf = u.cf.codifferential()
f[time].reduce()
# # f[0].visualize.quick()
#
msehtt_ls = obj['ls'].apply()
# msehtt_ls.pr()
# # print(msehtt_ls)
# # print(u.cf.field)
# #
msehtt_ls.config(['essential bc', 1], boundary_p, phi_scalar, root_form=phi)    # phi bc
msehtt_ls.config(('natural bc', 1), boundary_u, u.cf.field, root_form=u)    # u bc

linear_system = msehtt_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')
linear_system.x.update(x)
# print(phi[time].error(), u[time].error())
assert phi[time].error() < 0.0006
assert u[time].error() < 0.03
