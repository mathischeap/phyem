# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/Poisson2.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph
N = 3
K = 15
c = 0.1

time = 1

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='outer', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)('crazy', element_layout=K, c=c, bounds=([0, 1], [0, 1]), periodic=False)
# msehtt.config(tgm)('backward_step', element_layout=K)

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
        'ounv': ([-1, 0], [0, -1])    # outward unit norm vector.
    }
)
# boundary_p.visualize()

msehtt.config(boundary_u)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([1, 0], [0, 1], )    # outward unit norm vector.
    }
)
# boundary_u.visualize()

phi = msehtt.base['forms'][r'potential']
u = msehtt.base['forms'][r'velocity']
f = msehtt.base['forms'][r'source']


def phi_func(t, x, y):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t


phi_scalar = ph.vc.scalar(phi_func)

phi.cf = phi_scalar
# phi[time].reduce()
# phi[0].visualize.quick()

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[time].reduce()
f[0].reduce()

# f[time/2].cochain = f(time/2).cochain
# f[time/2].visualize()
# f[0].visualize.quick()

msehtt_ls = obj['ls'].apply()
# msehtt_ls.pr()
# print(msehtt_ls)
# print(u.cf.field)

msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc

linear_system = msehtt_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')

# x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)
# print(message)
# print(x)
linear_system.x.update(x)
# print(phi[time].error())

assert phi[time].error() < 0.0005
# phi[time].visualize()
#
assert u[time].error() < 0.002
# # u[time].visualize.quick()
#
# phi.saveto('phi.mse')
#
# phi.cochain.clean('all')
# phi.read('phi.mse')
# assert time in phi.cochain
# print(phi[time].error())

#
# ph.vtk('poisson_forms', phi[time], u[time], f[time])
# ph.os.remove('poisson_forms.vtu')
#
# u[time].export.rws('u_rws')
# dds = ph.read('u_rws')
#
# ph.os.remove('phi.mse')
#
# # if dds is not None:
# #     dds.visualize()
# #     sf = dds.streamfunction()
# #     sf.visualize()
#
# ph.os.remove('u_rws')
#
# p0x, p0y = u[time].project.to('m2n2k0')
#
# p0 = msehtt.base['forms'][r'helper0']
# p1 = msehtt.base['forms'][r'helper1']
#
# p0[time].cochain = p0x
# # p0[time].visualize()
#
# p1[time].cochain = p0[time].cochain.coboundary()
# tsp = p1.numeric.tsp.components()[0]
# tsp = - tsp
