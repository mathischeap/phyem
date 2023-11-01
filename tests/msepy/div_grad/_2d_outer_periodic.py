# -*- coding: utf-8 -*-
r"""
Here we demonstrate how to use *phyem*, the *msepy* implementation, to solve the two-dimensional div-grad problem.

In two-dimensions, the mixed formulation of the div-grad problem is

.. math::
    \begin{equation}\left\lbrace
    \begin{aligned}
        u ^1 &= \mathrm{d}^{\ast}\varphi^2 ,\\
        - \mathrm{d} u^1 &= f^2.
    \end{aligned}\right.
    \end{equation}

We use smooth manufactured solutions for this case. The exact solution for :math:`\varphi` is

.. math::
    \varphi = - \sin(2\pi x) \sin(2\pi y).

Exact solutions of :math:`u^1` and :math:`f^2` then follow.
We consider the domain to be :math:`\Omega = (x,y) \in [0,1]^2` and it is fully periodic.
We use the :ref:`GALLERY-msepy-domains-and-meshes=multi-crazy` for this test. The solver is given below.


.. autofunction:: tests.msepy.div_grad._2d_outer_periodic.div_grad_2d_periodic_manufactured_test

========
Examples
========

Below, we use mimetic spectral elements of degree 2 on a uniform mesh of :math:`4 * 4 * 4` :math:`(K=4)` elements.

>>> errors4 = div_grad_2d_periodic_manufactured_test(2, 4)
>>> errors4[0]  # doctest: +ELLIPSIS
0.01...

We increase :math:`K` to :math:`K=8`, we do

>>> errors8 = div_grad_2d_periodic_manufactured_test(2, 8)

We can compute the convergence rate of the :math:`L^2`-error of solution :math:`\varphi_h^2` by

>>> import numpy as np
>>> rate = (np.log10(errors4[0]) - np.log10(errors8[0])) / (np.log10(1/4) - np.log10(1/8))
>>> round(rate, 1)
2.0

The optimal convergence rate is obtained.

"""


def phi_func(t, x, y):
    """"""
    return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + t * 0


import sys

ph_dir = '../'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import numpy as np
import phyem as ph


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


if __name__ == '__main__':
    # python tests/msepy/div_grad/_2d_outer_periodic.py
    import doctest
    doctest.testmod()
    errors = div_grad_2d_periodic_manufactured_test(2, 8)
    print(errors)
