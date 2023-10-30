
.. testsetup:: *

    import __init__ as ph
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(True)
    manifold = ph.manifold(2)
    mesh = ph.mesh(manifold)

.. testcleanup::

    pass


.. _docs-space-form:

============
Space & form
============

.. topic:: Estimated reading time

    ⏱️ 10 minutes


.. automodule:: src.spaces.main
    :undoc-members:


.. automodule:: src.form.main
    :undoc-members:


|

↩️  Back to :ref:`Documentation`.
