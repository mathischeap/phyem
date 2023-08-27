# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import new

_implemented_space_operators = (
    'wedge',
    'Hodge',
    'd',
    'codifferential',
    'trace',
)


def wedge(s1, s2):
    """"""
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace
    from src.spaces.continuous.bundle import BundleValuedFormSpace
    assert s1.mesh == s2.mesh, f"two entries have different meshes."

    if s1.__class__ is ScalarValuedFormSpace and s2.__class__ is ScalarValuedFormSpace:

        k = s1.k
        l_ = s2.k

        assert k + l_ <= s1.mesh.ndim

        o1, o2 = s1.orientation, s2.orientation

        if o1 == o2:
            orientation = o1

        else:
            orientation = None

        return new('Lambda', k + l_, mesh=s1.mesh, orientation=orientation)

    elif s1.__class__ is BundleValuedFormSpace and s2.__class__ is BundleValuedFormSpace:

        k = s1.k
        l_ = s2.k

        assert k + l_ <= s1.mesh.ndim

        o1, o2 = s1.orientation, s2.orientation

        if o1 == o2:
            orientation = o1

        else:
            orientation = None   # unknown orientation.

        return new('bundle', k + l_, mesh=s1.mesh, orientation=orientation)

    else:
        raise NotImplementedError()


def tensor_product(s1, s2):
    """"""
    from src.spaces.continuous.bundle import BundleValuedFormSpace

    if s1.__class__ is BundleValuedFormSpace and s2.__class__ is BundleValuedFormSpace:

        assert s1.mesh == s2.mesh, f"two entries have different meshes."

        m, n = s1.mesh.m, s1.mesh.n

        if m == n == 2:  # on 2d mesh in 2d space.
            k1 = s1.k
            k2 = s2.k

            if k1 == 0 and k2 == 0 and s1.orientation == 'inner' and s2.orientation == 'inner':
                return new('bundle', 1, mesh=s1.mesh, orientation='inner')
            else:
                raise NotImplementedError()

    else:
        raise NotImplementedError()


def cross_product(s1, s2):
    """"""
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace

    if s1.__class__ is ScalarValuedFormSpace and s2.__class__ is ScalarValuedFormSpace:

        assert s1.mesh == s2.mesh, f"two entries have different meshes."

        m, n = s1.mesh.m, s1.mesh.n
        if m == n == 2:  # on 2d mesh in 2d space.
            k1 = s1.k
            k2 = s2.k

            if k1 == 0 and k2 == 1 and s1.orientation == 'outer' and s2.orientation == 'inner':
                return new('Lambda', 1, mesh=s1.mesh, orientation='inner')
            elif k1 == 0 and k2 == 1 and s1.orientation == 'outer' and s2.orientation == 'outer':
                return new('Lambda', 1, mesh=s1.mesh, orientation='outer')
            elif k1 == 2 and k2 == 1 and s1.orientation == 'inner' and s2.orientation == 'outer':
                return new('Lambda', 1, mesh=s1.mesh, orientation='outer')
            elif k1 == 1 and k2 == 1 and s1.orientation == 'inner' and s2.orientation == 'inner':
                return new('Lambda', 0, mesh=s1.mesh, orientation='outer')
            elif k1 == 1 and k2 == 1 and s1.orientation == 'outer' and s2.orientation == 'outer':
                return new('Lambda', 2, mesh=s1.mesh, orientation='inner')
            else:
                raise NotImplementedError(k1, k2, s1.orientation, s2.orientation)

    else:
        raise NotImplementedError()


def Hodge(space):
    """A not well-defined one"""
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace
    from src.spaces.continuous.bundle import BundleValuedFormSpace
    if space.__class__ is ScalarValuedFormSpace:
        n = space.mesh.ndim
        return new('Lambda', n - space.k, mesh=space.mesh, orientation=space.opposite_orientation)
    elif space.__class__ is BundleValuedFormSpace:
        n = space.mesh.ndim
        return new('bundle', n - space.k, mesh=space.mesh, orientation=space.opposite_orientation)
    else:
        raise NotImplementedError()


def d(space):
    """the range of exterior derivative operator on `space`."""
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace
    from src.spaces.continuous.bundle import BundleValuedFormSpace
    if space.__class__ is ScalarValuedFormSpace:
        assert space.k < space.mesh.ndim, f'd of top-form-space: {space} is 0.'
        return new('Lambda', space.k + 1, mesh=space.mesh, orientation=space.orientation)
    elif space.__class__ is BundleValuedFormSpace:
        assert space.k < space.mesh.ndim, f'd of top-form-space: {space} is 0.'
        return new('bundle', space.k + 1, mesh=space.mesh, orientation=space.orientation)
    else:
        raise NotImplementedError()


def codifferential(space):
    """the range of exterior derivative operator on `space`."""
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace
    from src.spaces.continuous.bundle import BundleValuedFormSpace
    if space.__class__ is ScalarValuedFormSpace:
        assert space.k > 0, f'd of 0-form is 0.'
        return new('Lambda', space.k - 1, mesh=space.mesh, orientation=space.orientation)
    elif space.__class__ is BundleValuedFormSpace:
        assert space.k > 0, f'd of 0-form is 0.'
        return new('bundle', space.k - 1, mesh=space.mesh, orientation=space.orientation)
    else:
        raise NotImplementedError(f"codifferential of {space} is not implemented or not even possible.")


def _d_to_vc(space_indicator, *args):
    """The correspondence between exterior derivative and vector calculus operators.

    in 2d, for inner forms, de Rham complex is :
        grad -> rot

    for outer:
        curl -> div

    """
    if space_indicator in ('Lambda', 'bundle'):  # scalar valued form spaces.
        m, n, k, ori = args

        if m == n == 1 and k == 0:  # 0-form on 1d manifold in 1d space.
            return 'derivative'

        elif m == n == 2 and k == 0:
            if ori == 'inner':
                return 'gradient'
            elif ori == 'outer':
                return 'curl'
            else:
                raise Exception()

        elif m == n == 2 and k == 1:
            if ori == 'inner':
                return 'rot'
            elif ori == 'outer':
                return 'divergence'
            else:
                raise Exception()

        elif m == n == 3:
            if k == 0:
                return 'gradient'
            elif k == 1:
                return 'curl'
            elif k == 2:
                return 'divergence'
            else:
                raise Exception()
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()


def _d_ast_to_vc(space_indicator, *args):
    """The correspondence between codifferential and vector calculus operators.
    """
    if space_indicator in ('Lambda', 'bundle'):  # scalar valued form spaces.
        m, n, k, ori = args
        if m == n == 1 and k == 1:  # 1-form on 1d manifold in 1d space.
            return '-', 'derivative'

        elif m == n == 2 and k == 1:
            if ori == 'inner':
                return '-', 'divergence'
            elif ori == 'outer':
                return '+', 'rot'
            else:
                raise Exception()
        elif m == n == 2 and k == 2:
            if ori == 'inner':
                return '+', 'curl'
            elif ori == 'outer':
                return '-', 'gradient'
            else:
                raise Exception()

        elif m == n == 3:
            if k == 1:
                return '-', 'divergence'
            elif k == 2:
                return '+', 'curl'
            elif k == 3:
                return '-', 'gradient'
            else:
                raise Exception()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def trace(space):
    from src.spaces.continuous.Lambda import ScalarValuedFormSpace
    if space.__class__ is ScalarValuedFormSpace:
        mesh = space.mesh
        assert 0 <= space.k < mesh.ndim, f"Cannot do trace on {space}."
        boundary_mesh = mesh.boundary()
        return new('Lambda', space.k, mesh=boundary_mesh, orientation=space.orientation)

    else:
        raise NotImplementedError(f"trace of {space} is not implemented or not even possible.")
