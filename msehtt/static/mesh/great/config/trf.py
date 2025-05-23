# -*- coding: utf-8 -*-
r"""
"""

from msehtt.static.mesh.great.config.msehtt_trf import _parse_trf
from msehtt.static.mesh.great.config.msehtt_trf import _refining_
from msehtt.static.mesh.great.config.msehtt_trf import _make_element_map_m2n2_
from msehtt.static.mesh.great.config.msehtt_trf import _finalize_m2n2_


def TRF(
        BASE_Element_type_dict,
        BASE_Element_parameter_dict,
        BASE_Element_map_dict,
        trf = 0,
):
    r"""

    Returns
    -------

    """
    base_mesh = BASE_Element_type_dict, BASE_Element_parameter_dict, BASE_Element_map_dict

    rff, rft, rcm = _parse_trf(trf)

    if isinstance(trf, dict) and 'rcm' in trf:
        pass
    else:
        rcm = 'middle'

    mn, elements = _refining_(base_mesh, rff, rft, rcm)

    if mn == (2, 2):
        element_map = _make_element_map_m2n2_(base_mesh, elements)
    else:
        raise NotImplementedError()

    if mn == (2, 2):
        Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict = _finalize_m2n2_(
            base_mesh, element_map,
        )
    else:
        raise NotImplementedError()

    return Element_Type_Dict, Element_Parameter_Dict, Element_Map_Dict
