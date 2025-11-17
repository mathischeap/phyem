# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python tests/msehtt/Poisson2_outer_meshpy_ts1.py
"""
import numpy as np
import phyem as ph
from phyem.tools.functions.space._2d.Cartesian_polar_coordinates_switcher import CartPolSwitcher


def phi_func(t, x, y):
    """"""
    r, theta = CartPolSwitcher.cart2pol(x, y)
    return r ** (2/3) * np.sin((2/3) * (theta + np.pi / 2)) + 0 * t

time = 0

ls = ph.samples.wf_div_grad(n=2, degree=2, orientation='outer', periodic=False)[0]

msehtt, obj = ph.fem.apply('msehtt-s', locals())
tgm = msehtt.tgm()

msehtt.config(tgm)(
    'meshpy',
    points=(
        [0, 0],
        [0, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, 0],
    ),
    max_volume=0.01,
    ts=1,
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
        # 'ounv': ([-1, 0], [0, -1])    # outward unit norm vector.
        'on straight lines': ([(0, 0), (-1, 0)], [(0, 0), (0, -1)])
    }
)
# boundary_p.visualize()

msehtt.config(boundary_u)(
    tgm,
    including={
        'type': 'boundary_section',
        'partial elements': msehtt_mesh,
        # 'ounv': ([1, 0], [0, 1], )    # outward unit norm vector.
        'except on straight lines': ([(0, 0), (-1, 0)], [(0, 0), (0, -1)])
    }
)
# boundary_u.visualize()

phi = msehtt.base['forms'][r'potential']
u = msehtt.base['forms'][r'velocity']
f = msehtt.base['forms'][r'source']

phi_scalar = ph.vc.scalar(phi_func)

phi.cf = phi_scalar

u.cf = - phi.cf.codifferential()
# f.cf = ph.vc.scalar(0)
f.cf = -u.cf.exterior_derivative()
f[time].reduce()

# u[time].reduce()
# u[time].visualize()

msehtt_ls = obj['ls'].apply()

msehtt_ls.config(['natural bc', 1], boundary_p, phi_scalar, root_form=phi)    # natural bc
msehtt_ls.config(('essential bc', 1), boundary_u, u.cf.field, root_form=u)    # essential bc

linear_system = msehtt_ls(time)

Axb = linear_system.assemble()
x, message, info = Axb.solve('direct')
# x, message, info = Axb.solve('gmres', x0=[u, phi], restart=300, maxiter=5)

linear_system.x.update(x)

dofs_phi = phi.dofs.num_global_dofs
dofs_u = u.dofs.num_global_dofs

phi_L2_error = phi[time].error()
u_L2_error = u[time].error()
u_H1_error = u[time].error(error_type='H1')

DIFF = f + u.d()

d_error = DIFF[None].norm()

np.testing.assert_almost_equal(phi_L2_error, 0.0027079138355503847)
np.testing.assert_almost_equal(u_L2_error, 0.027229207249787675)
np.testing.assert_almost_equal(u_H1_error, 0.027338490102690755)
np.testing.assert_almost_equal(d_error, 1.2980724718259327e-14)
