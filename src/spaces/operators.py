# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
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
    if s1.__class__.__name__ == 'ScalarValuedFormSpace' and s2.__class__.__name__ == 'ScalarValuedFormSpace':

        assert s1.mesh == s2.mesh, f"two entries have different meshes."

        k = s1.k
        l_ = s2.k

        assert k + l_ <= s1.mesh.ndim

        o1, o2 = s1.orientation, s2.orientation

        if o1 == o2:
            orientation = o1
        else:
            orientation = None

        return new('Lambda', k + l_, mesh=s1.mesh, orientation=orientation)

    else:
        raise NotImplementedError()


def Hodge(space):
    """A not well-defined one"""
    if space.__class__.__name__ == 'ScalarValuedFormSpace':
        n = space.mesh.ndim
        return new('Lambda', n - space.k, mesh=space.mesh, orientation=space.orientation)
    else:
        raise NotImplementedError()


def d(space):
    """the range of exterior derivative operator on `space`."""
    if space.__class__.__name__ == 'ScalarValuedFormSpace':
        assert space.k < space.mesh.ndim, f'd of top-form-space: {space} is 0.'
        return new('Lambda', space.k + 1, mesh=space.mesh, orientation=space.orientation)
    else:
        raise NotImplementedError()


def codifferential(space):
    """the range of exterior derivative operator on `space`."""
    if space.__class__.__name__ == 'ScalarValuedFormSpace':
        assert space.k > 0, f'd of 0-form is 0.'
        return new('Lambda', space.k - 1, mesh=space.mesh, orientation=space.orientation)
    else:
        raise NotImplementedError(f"codifferential of {space} is not implemented or not even possible.")


def trace(space):
    if space.__class__.__name__ == 'ScalarValuedFormSpace':
        mesh = space.mesh
        assert 0 <= space.k < mesh.ndim, f"Cannot do trace on {space}."
        boundary_mesh = mesh.boundary()
        return new('Lambda', space.k, mesh=boundary_mesh, orientation=space.orientation)

    else:
        raise NotImplementedError(f"trace of {space} is not implemented or not even possible.")
