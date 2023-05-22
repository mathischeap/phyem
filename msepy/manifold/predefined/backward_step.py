# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
from msepy.manifold.predefined._helpers import _LinearTransformation


def backward_step(mf, x1=1, x2=1, y1=0.25, y2=0.25, z=None):
    """
    ^ y
    |
    |              x1                      x2
    |   __________________________________________________
    |   |                    |                           |
    |   |                    |                           |
    |   |         r2         |           r1              |    y2
    |   |                    |                           |
    |   |____________________|___________________________|
    |                        |                           |
    |                        |                           |
    |                        |           r0              |    y1
    |                        |                           |
    | (0,0)                  |___________________________|
    |
    .--------------------------------------------------------------> x


    Parameters
    ----------
    mf
    x1
    x2
    y1
    y2
    z

    Returns
    -------

    """
    assert mf.esd == mf.ndim, f"backward_step mesh only works for manifold.ndim == embedding space dimensions."
    assert mf.esd in (2, 3), f"backward_step mesh only works in 2-, 3-dimensions."
    esd = mf.esd
    if z is None:
        if esd == 2:
            z = 0
        else:
            z = 0.25
    else:
        pass
    if esd == 2:
        assert z == 0, f"for 2-d backward_step mesh, z must be 0."
    elif esd == 3:
        assert z > 0, f"for 3-d backward_step mesh, z must be greater than 0."
    else:
        raise NotImplementedError()

    if esd == 2:
        rm0 = _LinearTransformation(x1, x1+x2, 0,  y1)
        rm1 = _LinearTransformation(x1, x1+x2, y1, y1+y2)
        rm2 = _LinearTransformation(0,  x1,    y1, y1+y2)
    elif esd == 3:
        rm0 = _LinearTransformation(x1, x1+x2, 0,  y1,    0, z)
        rm1 = _LinearTransformation(x1, x1+x2, y1, y1+y2, 0, z)
        rm2 = _LinearTransformation(0,  x1,    y1, y1+y2, 0, z)
    else:
        raise Exception()

    if esd == 2:
        region_map = {
            0: [None, None, None, 1],
            1: [2,    None, 0,    None],
            2: [None, 1,    None, None],
        }
    elif esd == 3:
        region_map = {
            0: [None, None, None, 1,    None, None],
            1: [2,    None, 0,    None, None, None],
            2: [None, 1,    None, None, None, None],
        }
    else:
        raise Exception()

    mapping_dict = {
        0: rm0.mapping,  # region #0
        1: rm1.mapping,  # region #1
        2: rm2.mapping,  # region #2
    }

    Jacobian_matrix_dict = {
        0: rm0.Jacobian_matrix,  # region #1
        1: rm1.Jacobian_matrix,  # region #2
        2: rm2.Jacobian_matrix,  # region #3
    }

    if esd == 2:
        mtype_dict = {
            0: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y1}']},
            1: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y2}']},
            2: {'indicator': 'Linear', 'parameters': [f'x{x1}', f'y{y2}']},
        }
    elif esd == 3:
        mtype_dict = {
            0: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y1}', f'z{z}']},
            1: {'indicator': 'Linear', 'parameters': [f'x{x2}', f'y{y2}', f'z{z}']},
            2: {'indicator': 'Linear', 'parameters': [f'x{x1}', f'y{y2}', f'z{z}']}
        }
    else:
        raise Exception()

    return region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict
