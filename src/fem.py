# -*- coding: utf-8 -*-
"""
finite element setting

pH-lib@RAM-EEMCS-UT
created at: 3/30/2023 5:35 PM
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

from src.manifold import Manifold
from src.mesh import Mesh
from src.spaces.base import SpaceBase
from src.form.main import Form

from src.manifold import _global_manifolds  # [manifold_sym_repr] -> manifold
from src.mesh import _global_meshes  # [mesh_sym_repr] -> mesh
from src.spaces.main import _space_set  # [mesh_sym_repr][space_sym_repr] -> space
from src.form.main import _global_root_forms_lin_dict  # [root-form_lin_repr] -> root-form
import msepy.main as msepy


_implemented_finite_elements = {
    'msepy': msepy,   # mimetic spectral elements
}


_finite_elements_setup = dict()


def apply(fe_name, obj_dict):
    """"""
    assert fe_name in _implemented_finite_elements, \
        f"finite element name={fe_name} is wrong, should be one of {_implemented_finite_elements.keys()}"

    implementation = _implemented_finite_elements[fe_name]
    implementation._check_config()
    implementation._parse_manifolds(_global_manifolds)  # important, for all manifolds
    implementation._parse_meshes(_global_meshes)        # important, for all meshes
    implementation._parse_spaces(_space_set)            # important, for all spaces
    implementation._parse_root_forms(_global_root_forms_lin_dict)   # important, for all root-forms

    obj_space = dict()
    for obj_name in obj_dict:
        obj = obj_dict[obj_name]
        particular_obj = _parse_obj(implementation, obj)
        if particular_obj is not None:
            obj_space[obj_name] = particular_obj
        else:
            pass

    return implementation, obj_space


def _parse_obj(implementation, obj):
    """"""
    if obj.__class__ is Manifold:
        return implementation.base['manifolds'][obj._sym_repr]
    elif obj.__class__ is Mesh:
        return implementation.base['meshes'][obj._sym_repr]
    elif issubclass(obj.__class__, SpaceBase):
        if obj._sym_repr in implementation.base['spaces']:
            return implementation.base['spaces'][obj._sym_repr]
        else:
            pass  # for those spaces have no particular counterparts, we simply skip them.
    elif obj.__class__ is Form:
        if obj.is_root():
            if obj._lin_repr in implementation.base['forms']:
                return implementation.base['forms'][obj._lin_repr]
            else:
                pass  # for those spaces have no particular counterparts, we simply skip them.
        else:
            pass  # non-root-form has no counterpart.
    else:
        return implementation._parse(obj)


if __name__ == '__main__':
    # python src/fem.py
    import __init__ as ph

    samples = ph.samples

    periodic = False
    oph = samples.pde_canonical_pH(n=3, p=3, periodic=periodic)[0]
    a3, b2 = oph.unknowns
    # oph.pr()

    wf = oph.test_with([a3, b2], sym_repr=[r'v^3', r'u^2'])

    wf = wf.derive.integration_by_parts('1-1')
    # wf.pr(indexing=True)
    if periodic is False:

        td = wf.td
        td.set_time_sequence()  # initialize a time sequence

        td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
        td.differentiate('0-0', 'k-1', 'k')
        td.average('0-1', b2, ['k-1', 'k'])

        td.differentiate('1-0', 'k-1', 'k')
        td.average('1-1', a3, ['k-1', 'k'])
        td.average('1-2', a3, ['k-1/2'])
        dt = td.time_sequence.make_time_interval('k-1', 'k')

        wf = td()

        # wf.pr()

        wf.unknowns = [
            a3 @ td.time_sequence['k'],
            b2 @ td.time_sequence['k'],
        ]

        wf = wf.derive.split(
            '0-0', 'f0',
            [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
            ['+', '-'],
            factors=[1/dt, 1/dt],
        )

        wf = wf.derive.split(
            '0-2', 'f0',
            [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
            ['+', '+'],
            factors=[1/2, 1/2],
        )

        wf = wf.derive.split(
            '1-0', 'f0',
            [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
            ['+', '-'],
            factors=[1/dt, 1/dt]
        )

        wf = wf.derive.split(
            '1-2', 'f0',
            [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
            ['+', '+'],
            factors=[1/2, 1/2],
        )

        wf = wf.derive.rearrange(
            {
                0: '0, 3 = 2, 1',
                1: '3, 0 = 2, 1, 4',
            }
        )

        ph.space.finite(3)

        mp = wf.mp()
        # mp.parse([
        #     a3 @ td.time_sequence['k-1'],
        #     b2 @ td.time_sequence['k-1']]
        # )
        ls = mp.ls()

    else:

        td = wf.td
        td.set_time_sequence()  # initialize a time sequence

        td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
        td.differentiate('0-0', 'k-1', 'k')
        td.average('0-1', b2, ['k-1', 'k'])

        td.differentiate('1-0', 'k-1', 'k')
        td.average('1-1', a3, ['k-1', 'k'])
        dt = td.time_sequence.make_time_interval('k-1', 'k')

        wf = td()

        # wf.pr()

        wf.unknowns = [
            a3 @ td.time_sequence['k'],
            b2 @ td.time_sequence['k'],
            ]

        wf = wf.derive.split(
            '0-0', 'f0',
            [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
            ['+', '-'],
            factors=[1/dt, 1/dt],
        )

        wf = wf.derive.split(
            '0-2', 'f0',
            [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
            ['+', '+'],
            factors=[1/2, 1/2],
        )

        wf = wf.derive.split(
            '1-0', 'f0',
            [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
            ['+', '-'],
            factors=[1/dt, 1/dt]
        )

        wf = wf.derive.split(
            '1-2', 'f0',
            [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
            ['+', '+'],
            factors=[1/2, 1/2],
        )

        wf = wf.derive.rearrange(
            {
                0: '0, 3 = 2, 1',
                1: '3, 0 = 2, 1',
            }
        )

        ph.space.finite(3)

        mp = wf.mp()
        # mp.parse([
        #     a3 @ td.time_sequence['k-1'],
        #     b2 @ td.time_sequence['k-1']]
        # )
        ls = mp.ls()

    # mp.pr()
    # ls.pr()
    mesh = oph.mesh
    manifold = mesh.manifold

    a3k = a3 @ td.ts['k']
    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    msepy.config(mnf)('backward_step')
    msepy.config(msh)([3, 5, 3])

    a = obj['a3']
    b = obj['b2']
    ak = obj['a3k']

    ak[1].cochain = 100
    print(a[1].cochain)
