# -*- coding: utf-8 -*-
r"""
"""

import sys
ph_dir = '../../'  # customize it to your own dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import phyem as ph
import numpy as np


def div_grad_2d_general_bc_manufactured_test(degree, K, c=0.):
    r"""

    Parameters
    ----------
    degree : int
        The degree of the mimetic spectral elements.
    K : int
        In total we will use :math:`4 * K * K` elements.
    c : float, default=0
        The deformation factor of the :ref:`GALLERY-msepy-domains-and-meshes=crazy`.

    Returns
    -------
    phi_error: float
        The :math:`L^2`-error of solution :math:`\varphi_h^2`.
    u_error: float
        The :math:`L^2`-error of solution :math:`u_h^1`.

    """

    n = 2
    ls, mp = ph.samples.wf_div_grad(n=n, degree=degree, orientation='outer', periodic=False)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    boundary_manifold = msepy.base['manifolds'][r"\partial\mathcal{M}"]
    Gamma_phi = msepy.base['manifolds'][r"\Gamma_\phi"]
    Gamma_u = msepy.base['manifolds'][r"\Gamma_u"]

    msepy.config(manifold)(
        'crazy', c=c, bounds=[[0., 1.] for _ in range(n)], periodic=False,
    )

    msepy.config(Gamma_u)(
        manifold, {0: [1, 1, 0, 0]}
    )

    # manifold.visualize()
    # boundary_manifold.visualize()
    # Gamma_phi.visualize()
    # Gamma_u.visualize()

    mesh = msepy.base['meshes'][r'\mathfrak{M}']
    msepy.config(mesh)([K, K])

    # for mesh_repr in msepy.base['meshes']:
    #     mesh = msepy.base['meshes'][mesh_repr]
    #     mesh.visualize()

    phi = msepy.base['forms']['potential']
    u = msepy.base['forms']['velocity']
    f = msepy.base['forms']['source']

    ls = obj['ls'].apply()

    def phi_func(t, x, y):
        """"""
        return - np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + t * 0

    phi_scalar = ph.vc.scalar(phi_func)

    phi.cf = phi_scalar
    u.cf = - phi.cf.codifferential()
    f.cf = - u.cf.exterior_derivative()
    # u.cf = phi_scalar.gradient
    # f.cf = - phi_scalar.gradient.divergence

    ls.bc.config(Gamma_phi)(phi_scalar)
    ls.bc.config(Gamma_u)(phi_scalar.gradient)

    f[0].reduce()
    # phi[0].reduce()
    # f[0].visualize()

    ls0 = ls(0)
    als = ls0.assemble()
    results = als.solve()
    ls0.x.update(results[0])

    # phi[0].visualize()
    # u[0].visualize()

    return phi[0].error(), u[0].error()
