# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/adaptive/Poisson2.py
"""

import numpy as np

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 3

time = 1

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='outer', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-a', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'chaotic', element_layout=2, c=0, bounds=([0, 1], [0, 1]), periodic=False
)

msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
msehtt.config(msehtt_mesh)(tgm, including='all')

total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

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

msehtt.config(boundary_u)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        'ounv': ([1, 0], [0, 1], )    # outward unit norm vector.
    }
)
msehtt.initialize()

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

msehtt_ls = obj['ls']
msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc

f[time].reduce()

msehtt.renew(trf=1)

# f[time].visualize()
# u[time].visualize()

static_ls = msehtt_ls.apply()  # whenever the implementation is renewed, re-apply ths system.

linear_system = static_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')

linear_system.x.update(x)

# print(phi[time].error())
# print(u[time].error())

assert phi[time].error() < 0.0006
assert u[time].error() < 0.0015
