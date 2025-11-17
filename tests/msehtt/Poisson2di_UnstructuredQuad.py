# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/Poisson2di_UnstructuredQuad.py
"""
import numpy as np

import phyem as ph
N = 3
K = 7
c = 0.

time = 1

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='inner', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy',
    element_layout=K, c=c, bounds=([0.25, 1.25], [0.25, 1.25]),
    periodic=False, trf=1, ts=1
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

u.cf = phi.cf.exterior_derivative()
f.cf = u.cf.codifferential()
f[time].reduce()

msehtt_ls = obj['ls'].apply()

msehtt_ls.config(['essential bc', 1], boundary_p, phi_scalar, root_form=phi)    # essential bc
msehtt_ls.config(('natural bc', 1), boundary_u, u.cf.field, root_form=u)        # natural bc

linear_system = msehtt_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')
# x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)

linear_system.x.update(x)
assert phi[time].error() < 5e-6
assert u[time].error() < 0.0009
