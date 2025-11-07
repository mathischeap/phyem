# -*- coding: utf-8 -*-
r"""
"""


def bmatrix_to_array(text, column):
    r""""""
    matrix_begin_text = r"\left[\begin{array}{" + r"c" * column + r"}"
    matrix_end_text = r"\end{array}\right]"
    if r"\begin{bmatrix}" in text:
        res_text = text.replace(r"\begin{bmatrix}", matrix_begin_text)
    else:
        res_text = text

    if r"\end{bmatrix}" in res_text:
        res_text = res_text.replace(r"\end{bmatrix}", matrix_end_text)
    else:
        pass

    return res_text
