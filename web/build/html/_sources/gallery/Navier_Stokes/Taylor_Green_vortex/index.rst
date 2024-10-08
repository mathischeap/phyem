
.. _Gallery-NS2-TGV:

===================
Taylor-Green vortex
===================

The Taylor-Green vortex (TGV) is a series of analytical solutions of NS equations.
In 2-dimensions, the TGV analytical solutions are usually of the form,

.. math::
    \begin{aligned}
        u(x, y, t) &= - \sin(\pi x) \cos(\pi y) e^{-2\pi^2 t /\mathrm{Re}},\\
        v(x, y, t) &= \cos(\pi x) \sin(\pi y) e^{-2\pi^2 t /\mathrm{Re}},\\
        p(x, y, t) &= \frac{1}{4} \left(\cos(2\pi x) + \cos(2\pi y)\right)e^{-4\pi^2 t /\mathrm{Re}},\\
        \omega(x, y, t) &= -2\pi \sin(\pi x)\sin(\pi y) e^{-2\pi^2 t /\mathrm{Re}}.
    \end{aligned}

The domain, either periodic or not, is typically given as :math:`\Omega=[0,2]^2`. The above analytical solutions
are used for initial and boundary conditions. The simulation, for example, runs from :math:`t=0` to :math:`t=1` and
errors are measured at :math:`t=1`.

For a *phyem* implementation of the shear layer rollup using the dual-field method introduced in
`[Dual-field NS, Zhang et al., JCP (2022)] <https://doi.org/10.1016/j.jcp.2021.110868>`_, click
:download:`phyem_df2_TGV.py <../../../../../tests/msepy/dualNS2/TGV.py>`.

|

↩️  Back to :ref:`GALLERY-NS`.
