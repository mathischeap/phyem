# -*- coding: utf-8 -*-
r"""
"""
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from tools.vtk_.msehtt_form_static_copy import ___ph_vtk_msehtt_static_copy___


def vtk(filename, *args, **kwargs):
    """"""
    if all([_.__class__ is MseHttFormStaticCopy for _ in args]):
        ___ph_vtk_msehtt_static_copy___(filename, *args, **kwargs)
    else:
        raise NotImplementedError()
