
.. _GALLERY-MHD:

==========================
MHD (magnetohydrodynamics)
==========================

    **Magnetohydrodynamics** (**MHD**; also called **magneto-fluid dynamics** or **hydromagnetics**) is a model of
    electrically conducting fluids that treats all interpenetrating particle species together as a single continuous
    medium. It is primarily concerned with the low-frequency, large-scale, magnetic behavior in plasmas and liquid
    metals and has applications in numerous fields including geophysics, astrophysics, and engineering.

    -- wikipedia

Incompressible MHD
==================

In a connected, bounded domain :math:`\Omega \subset \mathbb{R}^{d}`, :math:`d\in\left\lbrace2,3\right\rbrace` with
Lipschitz boundary :math:`\partial \Omega`, the *incompressible constant density magnetohydrodynamic*
(or *simply incompressible MHD*) equations are given as

.. math::
    :label: incompressibleMHD

    \begin{equation}
    \begin{aligned}
        \rho \left[ \partial_t\boldsymbol{u}^* + \left(\boldsymbol{u}^* \cdot \nabla\right)\boldsymbol{u}^* \right]
        - \tilde{\mu} \Delta \boldsymbol{u}^*  - \boldsymbol{j}^* \times \boldsymbol{B}^* + \nabla p^*
        &= \rho \boldsymbol{f}^*, \\
        \nabla\cdot \boldsymbol{u}^* &= 0 ,\\
        \partial_t \boldsymbol{B}^* + \nabla\times \boldsymbol{E}^* &= \boldsymbol{0}  ,\\
        \boldsymbol{j}^* - \sigma \left(\boldsymbol{E}^* + \boldsymbol{u}^*\times\boldsymbol{B}^*\right) &= \boldsymbol{0} , \\
        \boldsymbol{j}^* - \nabla\times \boldsymbol{H}^* &= \boldsymbol{0} ,\\
        \boldsymbol{B}^* &= \mu \boldsymbol{H}^*,
    \end{aligned}
    \end{equation}

where

- :math:`\boldsymbol{u}^*` fluid velocity
- :math:`\boldsymbol{j}^*` electric current density
- :math:`\boldsymbol{B}^*` magnetic flux density
- :math:`p^*` hydrodynamic pressure
- :math:`\boldsymbol{f}^*` body force
- :math:`\boldsymbol{E}^*` electric field strength
- :math:`\boldsymbol{H}^*` magnetic field strength

subject to material parameters the fluid density :math:`\rho`, the dynamic viscosity :math:`\tilde{\mu}`,
the electric conductivity :math:`\sigma`, and the magnetic permeability :math:`\mu`.

By selecting the characteristic quantities of
length :math:`L`,
velocity :math:`U`,
and magnetic flux density :math:`B`,
a non-dimensional formulation of :eq:`incompressibleMHD` is

.. math::
    :label: non-dimensional-MHD

    \begin{equation}
    \begin{aligned}
        \partial_t\boldsymbol{u} + \left(\boldsymbol{u} \cdot \nabla\right)\boldsymbol{u}
        - \mathrm{R}_f^{-1} \Delta \boldsymbol{u}  - \mathrm{A}_l^{-2}\boldsymbol{j} \times \boldsymbol{B} + \nabla p
        &= \boldsymbol{f}, \\
        \nabla\cdot \boldsymbol{u} &= 0 ,\\
        \partial_t \boldsymbol{B} + \nabla\times \boldsymbol{E} &= \boldsymbol{0}  ,\\
        \mathrm{R}_m^{-1}\boldsymbol{j} - \left(\boldsymbol{E} + \boldsymbol{u}\times\boldsymbol{B}\right) &= \boldsymbol{0} , \\
        \boldsymbol{j} - \nabla\times \boldsymbol{B} &= \boldsymbol{0} ,\\
    \end{aligned}
    \end{equation}

where :math:`\boldsymbol{u}`, :math:`\boldsymbol{j}`, :math:`\boldsymbol{B}`, :math:`p`, :math:`\boldsymbol{f}`, and
:math:`\boldsymbol{E}` are the non-dimensional variables, and
:math:`\mathrm{R}_f = \dfrac{\rho U L}{\tilde{\mu}} = \dfrac{U L}{\nu}`
(with :math:`\nu=\dfrac{\tilde{\mu}}{\rho}`
being the kinematic viscosity) is the fluid Reynolds number,
:math:`\mathrm{A}_l = \dfrac{U\sqrt{\rho \mu}}{B} = \dfrac{U}{U_A}`
(with :math:`U_A = \dfrac{B}{\sqrt{\rho\mu}}` being the Alfvén speed), and
:math:`\mathrm{R}_m = \mu\sigma U L` is the magnetic Reynolds number.


If we further introduce :math:`\boldsymbol{\omega}:=\nabla\times\boldsymbol{u}` and
:math:`P:=p+\frac{1}{2}\boldsymbol{u}\cdot \boldsymbol{u}`, :eq:`non-dimensional-MHD` can be written into the
rotational form:

.. math::
    :label: rotational-non-dimensional-MHD

    \begin{equation}
    \begin{aligned}
        \partial_t\boldsymbol{u} + \boldsymbol{\omega}\times\boldsymbol{u}
        - \mathrm{R}_f^{-1} \Delta \boldsymbol{u}  - \mathrm{A}_l^{-2}\boldsymbol{j} \times \boldsymbol{B} + \nabla P
        &= \boldsymbol{f}, \\
        \boldsymbol{\omega} - \nabla\times\boldsymbol{u} &= \boldsymbol{0} ,\\
        \nabla\cdot \boldsymbol{u} &= 0 ,\\
        \partial_t \boldsymbol{B} + \nabla\times \boldsymbol{E} &= \boldsymbol{0}  ,\\
        \mathrm{R}_m^{-1}\boldsymbol{j} - \left(\boldsymbol{E} + \boldsymbol{u}\times\boldsymbol{B}\right) &= \boldsymbol{0} , \\
        \boldsymbol{j} - \nabla\times \boldsymbol{B} &= \boldsymbol{0} .\\
    \end{aligned}
    \end{equation}


Numerical Examples
==================
For numerical examples of MHD, see

.. toctree::
   :maxdepth: 1

   Orszag_Tang_vortex/index


|

↩️  Back to :ref:`GALLERY-Gallery`.
