
.. _GALLERY-MHD:

======
🧲 MHD
======

    **MHD (Magnetohydrodynamics)** (also called **magneto-fluid dynamics** or **hydromagnetics**) is a model of
    electrically conducting fluids that treats all interpenetrating particle species together as a single continuous
    medium. It is primarily concerned with the low-frequency, large-scale, magnetic behavior in plasmas and liquid
    metals and has applications in numerous fields including geophysics, astrophysics, and engineering.

    -- wikipedia

Incompressible MHD
==================

In a connected, bounded domain :math:`\Omega \subset \mathbb{R}^{d}`, :math:`d\in\left\lbrace2,3\right\rbrace`,
the *incompressible constant density magnetohydrodynamic* (or *simply incompressible MHD*) equations are given as

.. math::
    :label: incompressibleMHD

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

|

↩️  Back to :ref:`GALLERY-Gallery`.
