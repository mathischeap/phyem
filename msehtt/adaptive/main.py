# -*- coding: utf-8 -*-
r"""
"""
base = {
    'manifolds': dict(),  # keys: abstract manifold sym_repr
    'the_top_mesh': None,   # all forms/meshes will also point to this great mesh.
    'the_top_space': None,
    'the_top_form': None,
}


from msehtt.static.manifold.main import MseHttManifold
from msehtt.adaptive.mesh.main import MseHtt_Adaptive_TopMesh
from msehtt.adaptive.space.main import MseHtt_Adaptive_TopSpace
from msehtt.adaptive.form.main import MseHtt_Adaptive_TopForm

from src.wf.mp.linear_system import MatrixProxyLinearSystem
# from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem

from msehtt.adaptive.tools.linear_system import MseHtt_Adaptive_LinearSystem


def _check_config():
    """Check whether the configuration is compatible or not. And if necessary, prepare the data!"""


def _clear_self():
    """Clear self to make sure the previous implementation does not mess things up!"""
    base['manifolds'] = dict()
    base['the_top_mesh'] = None
    base['the_top_space'] = None
    base['the_top_form'] = None


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MseHttManifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    assert base['the_top_mesh'] is None, f"We must do not generate the top mesh yet."
    ttm = MseHtt_Adaptive_TopMesh(abstract_meshes)
    # noinspection PyTypedDict
    base['the_top_mesh'] = ttm


def _parse_spaces(abstract_spaces):
    r""""""
    assert base['the_top_space'] is None, f"We must do not generate the top mesh yet."
    tts = MseHtt_Adaptive_TopSpace(abstract_spaces)
    # noinspection PyTypedDict
    base['the_top_space'] = tts


def _parse_root_forms(abstract_rfs):
    r""""""
    assert base['the_top_form'] is None, f"We must do not generate the top mesh yet."
    ttf = MseHtt_Adaptive_TopForm(abstract_rfs)
    # noinspection PyTypedDict
    base['the_top_form'] = ttf


def _parse(obj):
    r"""The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        dynamic = MseHtt_Adaptive_LinearSystem(obj, base)
        return dynamic
    else:
        return None
