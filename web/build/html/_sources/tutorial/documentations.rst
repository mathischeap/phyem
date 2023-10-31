
.. _Documentation:

Documentations
==============

The documentations are presented in a manner reflecting the flow of a *phyem* simulation.
:ref:`Jupyter-notebooks` and :ref:`GALLERY-Gallery`  are complementary to them.

.. caution::

    A most complete *phyem* simulation would start with inputs defining a problem as a bond graph.
    However, the bond graph class, as well as its interface to the PDE class, is not implemented yet.
    As a compromise, the current documentation illustrates a problem defining through the PDE class.

    Since the bond graph class and its interface to the
    PDE class are key for the **LEGO**-like feature of *phyem*. Since they are missing, this feature is not
    realized for the time being.

Below we present documentations pf *phyem*.
They demonstrate a *phyem* simulation for a linear port-Hamiltonian problem.
It services as a good clue to understand the overall skeleton of a *phyem* simulation.
Examples for other problems can be
found in :ref:`Jupyter-notebooks` and :ref:`GALLERY-Gallery`. Nevertheless, we recommend users to start with
below documentations no matter which type of problem you are targeting at.


.. admonition:: Documentations

    A Python script that carries out a *phyem* simulation usually consists of commands for the
    following functionalities:

    .. toctree::
        :maxdepth: 2
        :numbered:

        docs/presetting
        docs/manifold_mesh
        docs/space_form
        docs/pde
        docs/wf
        docs/ap
        docs/implementations


More examples can be found at :ref:`Jupyter-notebooks` and :ref:`GALLERY-Gallery`.

|

↩️  Back to :ref:`PHYEM`.
