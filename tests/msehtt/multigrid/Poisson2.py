# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/multigrid/Poisson2.py
"""
import numpy as np
import phyem as ph

N = 3
K = 5
c = 0.

time = 2

ls = ph.samples.wf_div_grad(n=2, degree=N, orientation='outer', periodic=False)[0]
# ls.pr()

msehtt, obj = ph.fem.apply('msehtt-smg', locals())
tgm = msehtt.tgm()
msehtt.config(tgm)(
    'crazy', element_layout=K, c=c, bounds=([0, 1], [0, 1]), periodic=False,
    mgc=2,
)
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
phi[time].reduce()

u.cf = - phi.cf.codifferential()
f.cf = - u.cf.exterior_derivative()
f[time].reduce()
LS = obj['ls']
msehtt_ls = LS.apply()
msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc
linear_system = msehtt_ls(time)
Axb = linear_system.assemble()
x, message, info = Axb.solve('direct', inner_m=350, outer_k=15, maxiter=100)
linear_system.x.update(x)

p_error = phi[time].error()
u_error = u[time].error()
# print(p_error, u_error)
assert p_error < 0.0006
assert u_error < 0.004
