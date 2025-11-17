# -*- coding: utf-8 -*-
r"""Save objects to vtk file.
"""
from phyem.msehtt.static.form.addons.static import MseHttFormStaticCopy
from phyem.tools.vtk_.msehtt_form_static_copy import ___ph_vtk_msehtt_static_copy___


def vtk(filename, *args, **kwargs):
    """"""
    if '.' in filename:   # if we provide extension name, remove it since the package we use do not need it!
        if filename[-4:] == '.vtu':
            filename = filename[:-4]
        elif filename[-4:] == '.vtk':
            filename = filename[:-4]
        else:
            raise Exception(f"extension name must be .vtu or .vtk")
    else:
        pass

    if all([_.__class__ is MseHttFormStaticCopy for _ in args]):
        ___ph_vtk_msehtt_static_copy___(filename, *args, **kwargs)
    else:
        raise NotImplementedError()
