
.. testsetup:: *

    import __init__ as ph
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(False)

.. testcleanup::

    pass


.. _docs-manifold-mesh:

===============
Manifold & mesh
===============

.. topic:: Estimated reading time

    ⏱️ 5 minutes


.. automodule:: src.manifold
    :undoc-members:


.. automodule:: src.mesh
    :undoc-members:

.. note::

    Since so far everything is at the abstract level, we cannot visualize the manifold (i.e. the computational domain)
    or the mesh.

    We have the freedom to further define the manifold to be an exact one of particular size,
    shape, etc., and to define the mesh to be an exact one of certain amount of triangulated or quadrilateral
    cells (elements). These processes will be done when we invoke a particular implementation.

|

.. topic:: Python script of this section

    :download:`mm.py <py/mm.py>`

↩️  Back to :ref:`Documentation`.
