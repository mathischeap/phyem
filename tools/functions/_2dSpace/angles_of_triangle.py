# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 10/6/2022 5:36 PM
"""
from tools.functions._2dSpace.distance import distance
import math


def angles_of_triangle(A, B, C):

    # Square of lengths be a2, b2, c2
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)

    # length of sides be a, b, c
    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    beta = math.acos((a2 + c2 - b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))

    # Converting to degree
    alpha = alpha * 180 / math.pi
    beta = beta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    return alpha, beta, gamma
