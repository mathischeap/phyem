# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:21 PM on 5/10/2023
"""
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting
from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_d_matrix_reprs
from src.spaces.main import _default_d_matrix_transpose_reprs
from msepy.main import base

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)


def msepy_root_array_parser(array_lin_repr):
    """"""
    assert array_lin_repr[:_len_front] == _front and array_lin_repr[-_len_back:] == _back, \
        f"array_lin_repr={array_lin_repr} is not representing a root-array."
    array_lin_repr = array_lin_repr[_len_front:-_len_back]
    indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
    type_indicator = indicators[0]   # this first one indicates the type
    info_indicators = indicators[1:]  # the others indicate the details.

    if type_indicator == _default_mass_matrix_reprs[1].split(_sep)[0]:
        return _parse_mass_matrix(*info_indicators)
    elif type_indicator == _default_d_matrix_reprs[1].split(_sep)[0]:
        pass
    elif type_indicator == _default_d_matrix_transpose_reprs[1].split(_sep)[0]:
        pass
    else:
        raise NotImplementedError()


def _parse_mass_matrix(space, degree0, degree1):
    """"""
    print(space, degree0, degree1)



if __name__ == '__main__':
    # python 
    pass
