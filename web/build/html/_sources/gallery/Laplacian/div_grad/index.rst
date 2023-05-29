
.. _GALLERY-Laplacian-div-grad:

========
div-grad
========

Here we demonstrate how to use PHYEM to solve different types of Laplacian problems in different dimensions and
different domains.

The Laplacian equation is

.. math::

    -\mathrm{d} \mathrm{d}^{\ast} \varphi = f,

where :math:`f` is a :math:`k`-form, :math:`0 < k \leq n`.

When :math:`f` is a top form (outer-oriented in two-dimensions), it refers to a div-grad problem, or the so-called scalar Laplacian problem. For
demonstrations of the div-grad problem, see


.. automodule:: tests.unittests.msepy.div_grad.div_grad
    :undoc-members:
