

3d
--

.. testsetup:: *

    None_or_custom_path_2 = './source/gallery/msepy_domains_and_meshes/msepy/crazy_2d_c.png'
    import __init__ as ph

.. testcleanup::

    pass

Below codes generate a crazy domain in `\Omega:=(x,y,z)\in[-1,1]\times[0,2]\times[0,2]` of :math:`c=0.15`. A mesh
of :math:`5 * 5 * 5` elements are then generated in the domain ans is shown the following figure.

>>> ph.config.set_embedding_space_dim(3)
>>> manifold = ph.manifold(3)
>>> mesh = ph.mesh(manifold)
>>> msepy, obj = ph.fem.apply('msepy', locals())
>>> manifold = obj['manifold']
>>> mesh = obj['mesh']
>>> msepy.config(manifold)('crazy', c=0.15, periodic=False, bounds=[[-1, 1], [0, 2], [0, 2]])
>>> msepy.config(mesh)([5, 5, 5])
>>> mesh.visualize(saveto=None_or_custom_path_3)  # doctest: +ELLIPSIS
<Figure size ...

.. figure:: crazy_3d_c.png
    :height: 400

    The crazy mesh in :math:`\Omega=[-1,1]\times[0,2]\times[0,2]` of :math:`5 * 5 * 5` elements
    at deformation factor :math:`c=0.15`.