
.. _GALLERY-NS:

==========================
🌊 Navier-Stokes equations
==========================

    The **Navier–Stokes equations** (/nævˈjeɪ stoʊks/ nav-YAY STOHKS) are partial differential equations which
    describe the motion of viscous fluid substances, named after French engineer and physicist Claude-Louis Navier
    and Anglo-Irish physicist and mathematician George Gabriel Stokes. They were developed over several decades
    of progressively building the theories, from 1822 (Navier) to 1842-1850 (Stokes).

    -- wikipedia


Incompressibility
=================

In a connected, bounded domain :math:`\Omega \subset \mathbb{R}^{d}`, :math:`d\in\left\lbrace2,3\right\rbrace` with
a Lipschitz boundary :math:`\partial \Omega`, the incompressible (more strictly speaking, constant density)
Navier-Stokes equations are of the generic dimensionless form,

.. math::
    :label: generic-NS

    \begin{equation}
    \begin{aligned}
        \partial_t\boldsymbol{u} - \mathcal{C}(\boldsymbol{u}) - \mathrm{Re}^{-1}\mathcal{D}(\boldsymbol{u})
        +\nabla p &= \boldsymbol{f},\\
        \nabla\cdot\boldsymbol{u} &= 0,\\
    \end{aligned}
    \end{equation}

where :math:`\boldsymbol{u}` is the velocity field, :math:`p` is the static pressure,
:math:`\boldsymbol{f}` is the body force,
:math:`\mathrm{Re}` is the Reynolds number, :math:`\mathcal{C}(\boldsymbol{u})` and
:math:`\mathcal{D}(\boldsymbol{u})` represent the nonlinear convective term and the linear dissipative term,
respectively.


Numerical Examples
==================
For numerical simulations of Navier-Stokes flows with *phyem*, see

.. toctree::
    :maxdepth: 1

    backward_facing_step/index
    lid_driven_cavity/index
    normal_dipole_collision/index
    shear_layer_rollup/index
    Taylor_Green_vortex/index

|

↩️  Back to :ref:`GALLERY-Gallery`.
