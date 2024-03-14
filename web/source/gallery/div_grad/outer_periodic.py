# -*- coding: utf-8 -*-
r"""
"""

import sys
ph_dir = '../../'  # customize it to your own dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import phyem as ph
import numpy as np


def div_grad_2d_periodic_manufactured_test(degree, K, c=0):
    r"""

    Parameters
    ----------
    degree : int
        The degree of the mimetic spectral elements.
    K : int
        In total we will use :math:`4 * K * K` elements.
    c : float, default=0
        The deformation factor of the :ref:`GALLERY-msepy-domains-and-meshes=multi-crazy`.

    Returns
    -------
    phi_error: float
        The :math:`L^2`-error of solution :math:`\varphi_h^2`.
    u_error: float
        The :math:`L^2`-error of solution :math:`u_h^1`.

    """

    ls = ph.samples.wf_div_grad(n=2, degree=degree, orientation='outer', periodic=True)[0]  # ls.pr()
    msepy, obj = ph.fem.apply('msepy', locals())
    manifold = msepy.base['manifolds'][r"\mathcal{M}"]
    mesh = msepy.base['meshes'][r'\mathfrak{M}']

    msepy.config(manifold)(
        'crazy_multi',
        c=c,
        bounds=[
            [0, 1], [0, 1]
        ],
        periodic=True,
    )

    msepy.config(mesh)([K, K])

    phi = msepy.base['forms'][r'potential']
    u = msepy.base['forms'][r'velocity']
    f = msepy.base['forms'][r'source']

    ls = obj['ls'].apply()

    def phi_func(t, x, y):
        """"""
        return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0

    phi_scalar = ph.vc.scalar(phi_func)
    phi.cf = phi_scalar
    u.cf = - phi.cf.codifferential()
    f.cf = - u.cf.exterior_derivative()
    f[0].reduce()
    phi[0].reduce()

    ls0 = ls(0)
    ls0.customize.set_dof(-1, phi[0].cochain.of_dof(-1))
    als = ls0.assemble()
    x = als.solve()[0]
    ls0.x.update(x)

    phi_error = phi[0].error()
    u_error = u[0].error()

    return phi_error, u_error
