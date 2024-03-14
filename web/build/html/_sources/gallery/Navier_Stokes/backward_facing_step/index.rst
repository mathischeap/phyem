
.. _Gallery-NS2-BFS:

=========================
Backward facing step flow
=========================

The backward facing step flow is a 2-dimensional incompressible viscous flow.
The computational domain is illustrated in Fig. :any:`NS2-BFS-domain2`.

.. _NS2-BFS-domain2:

.. figure:: domain.png
    :width: 80 %

    Domain of the backward facing step flow

The flow enters the domain through the left edge of the domain and leaves the domain
through the right edge. Other edges are all no-slip solid walls. For example, we can
set the inlet velocity to be
:math:`\left.\boldsymbol{u}\right|_{\mathrm{in}}=\begin{bmatrix}1 & 0\end{bmatrix}^\mathsf{T}` and set the outlet
total pressure to be
:math:`\left.P\right|_{\mathrm{out}}=0`. On the solid walls, we just set
:math:`\left.\boldsymbol{u}\right|_{\mathrm{wall}} = \boldsymbol{0}`. See Fig.
:any:`NS2-BFS-snapshot`.

.. _NS2-BFS-snapshot:

.. figure:: velocity_vector_plot.png
    :width: 65 %

    A snapshot of the backward facing step flow.

For a *phyem* implementation of the backward facing step flow, click
:download:`phyem_bfs.py <../../../../../tests/msepy/MEEVC2/backward_facing_step.py>`

|

↩️  Back to :ref:`GALLERY-NS`.
