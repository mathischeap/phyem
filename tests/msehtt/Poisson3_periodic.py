# -*- coding: utf-8 -*-
"""
mpiexec -n 4 python tests/msehtt/Poisson3_periodic.py
"""

import numpy as np

import phyem as ph

N = 3
K = 5
c = 0.

ls = ph.samples.wf_div_grad(n=3, degree=N, orientation='outer', periodic=True)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K,
    c=c,
    bounds=([0.25, 1.25], [0.25, 1.25], [0.25, 1.25]),
    periodic=True
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')
# msehtt_mesh.visualize()

phi = msehtt.base['forms'][r'potential']
u = msehtt.base['forms'][r'velocity']
f = msehtt.base['forms'][r'source']


def phi_func(t, x, y, z):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 0


phi.cf = ph.vc.scalar(phi_func)
phi[0].reduce()
# print(phi[0].error())

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[0].reduce()
# print(f[0].error())

msehtt_ls = obj['ls'].apply()
# msehtt_ls.pr()

linear_system = msehtt_ls(0)
linear_system.customize.set_dof(-1, phi[0].cochain.of_dof(-1))
# # linear_system.customize.set_dof(-1, 1)

Axb = linear_system.assemble()
# Axb.spy()
# print(Axb.condition_number, Axb.rank, Axb.num_singularities, Axb.shape)
# print(Axb.shape)
#
x = Axb.solve('spsolve')[0]
# # print(x)
linear_system.x.update(x)
# # phi[0].visualize.quick()
# # u[0].visualize.quick()
# print(u[0].error())
# print(phi[0].error())
# ph.vtk('poisson', u[0], phi[0])
assert u[0].error() < 0.04
assert phi[0].error() < 0.004
