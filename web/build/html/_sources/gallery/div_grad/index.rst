

.. testsetup:: *

    from web.source.gallery.div_grad.outer_periodic import div_grad_2d_periodic_manufactured_test
    from web.source.gallery.div_grad.outer import div_grad_2d_general_bc_manufactured_test

.. testcleanup::

    pass


.. _GALLERY-Laplacian-div-grad:

===========
🟢 div-grad
===========



Here we demonstrate how to use *phyem* to solve the div-grad problems in different dimensions and
different domains.

The general form of the div-grad problem is

.. math::

    -\mathrm{d} \mathrm{d}^{\ast} \varphi^n = f^n,

where :math:`\varphi^n` and :math:`f^n` are top forms.



2d periodic boundary conditions
===============================

Here we demonstrate how to use *phyem*, the *msepy* implementation, to solve the two-dimensional div-grad problem.

In two-dimensions, the mixed formulation of the div-grad problem is

.. math::
    \left\lbrace
    \begin{aligned}
        u ^1 &= \mathrm{d}^{\ast}\varphi^2 ,\\
        - \mathrm{d} u^1 &= f^2.
    \end{aligned}\right.


We use smooth manufactured solutions for this case. The exact solution for :math:`\varphi` is

.. math::
    \varphi = - \sin(2\pi x) \sin(2\pi y).

Exact solutions of :math:`u^1` and :math:`f^2` then follow.
We consider the domain to be :math:`\Omega = (x,y) \in [0,1]^2` and it is fully periodic.
We use the :ref:`GALLERY-msepy-domains-and-meshes=multi-crazy` for this test. The solver is given below.

    .. autofunction:: web.source.gallery.div_grad.outer_periodic.div_grad_2d_periodic_manufactured_test

Examples
--------

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



2d general boundary conditions
==============================

Here we repeat the test, but with essential boundary :math:`\mathrm{tr}\ u^1`
on faces :math:`y=0` and :math:`y=1`, and natural boundary condition
:math:`\mathrm{tr}\left(\star \varphi^2\right)` on faces :math:`x=0` and :math:`x=1`.

The implementation is

    .. autofunction:: web.source.gallery.div_grad.outer.div_grad_2d_general_bc_manufactured_test


Examples
--------

If we solve it with :math:`4\times4` elements
(note that here we use a different mesh compared to the periodic test)
at polynomial degree 2,

>>> errors4 = div_grad_2d_general_bc_manufactured_test(2, 4)
>>> errors4[0]  # doctest: +ELLIPSIS
0.06...

We increase :math:`K` to :math:`K=8`, we do

>>> errors8 = div_grad_2d_general_bc_manufactured_test(2, 8)

We can compute the convergence rate of the :math:`L^2`-error of solution :math:`\varphi_h^2` by

>>> import numpy as np
>>> rate = (np.log10(errors4[0]) - np.log10(errors8[0])) / (np.log10(1/4) - np.log10(1/8))
>>> round(rate, 1)
2.0

Again, the optimal convergence rate is obtained.


|

↩️  Back to :ref:`GALLERY-Gallery`.
