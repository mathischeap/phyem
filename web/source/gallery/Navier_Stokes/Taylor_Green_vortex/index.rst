
.. _Gallery-NS2-TGV:

===================
Taylor-Green vortex
===================

The Taylor-Green vortex (TGV) is a series of analytical solutions of NS equations.
In 2-dimensions, the TGV analytical solution is usually of the form,

.. math::
    \begin{aligned}
        u(x, y, t) &= - \sin(\pi x) \cos(\pi y) e^{-2\pi^2 t /\mathrm{Re}},\\
        v(x, y, t) &= \cos(\pi x) \sin(\pi y) e^{-2\pi^2 t /\mathrm{Re}},\\
        p(x, y, t) &= \frac{1}{4} \left(\cos(2\pi x) + \cos(2\pi y)\right)e^{-4\pi^2 t /\mathrm{Re}},\\
        \omega(x, y, t) &= -2\pi \sin(\pi x)\sin(\pi y) e^{-2\pi^2 t /\mathrm{Re}}.
    \end{aligned}

The domain, either periodic or not, is given as :math:`\Omega=[0,2]^2`.

|

↩️  Back to :ref:`GALLERY-NS`.
