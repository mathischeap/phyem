# -*- coding: utf-8 -*-
r""""""
from tools.functions.space._2d.angle import angle


def if_two_lines_parallel(a1, a2, b1, b2):
    """
    Check if line p1-p2 is parallel with line p3-p4. If they are parallel but pointing different direction,
    return False

    :param a1:
    :param a2:
    :param b1:
    :param b2:
    :return:
    :rtype: bool
    """
    angle1 = angle(a1, a2)
    angle2 = angle(b1, b2)
    return angle1 == angle2
