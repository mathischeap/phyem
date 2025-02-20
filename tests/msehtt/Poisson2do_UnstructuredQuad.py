# -*- coding: utf-8 -*-
r"""
mpiexec -n 6 python tests/msehtt/Poisson2do_UnstructuredQuad.py
"""

import sys

import numpy as np

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import __init__ as ph
N = 2
K = 10
c = 0.

time = 1

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='outer', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'meshpy', points=[(0.125, 00.125), (1.125, 0.125), (1.125, 1.125), (0.125, 1.125)],
    max_volume=0.01, ts=1
)
# tgm.visualize()

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

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[time].reduce()
f[0].reduce()

msehtt_ls = obj['ls'].apply()

msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc

linear_system = msehtt_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')
# x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)

linear_system.x.update(x)
assert phi[time].error() < 0.01
assert u[time].error() < 0.03
