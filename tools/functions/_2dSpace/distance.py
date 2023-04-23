# -*- coding: utf-8 -*-
import numpy as np


def distance(p1, p2):
    """ Compute distance between two points. """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
