# -*- coding: utf-8 -*-
from tools.functions.space._2d.angle import angle
import numpy as np


class ArcAntiClockWise(object):
    """
    fit two center and radius int arc (up half) the return arc is ALWAYS
    anti-clock-wise!!
    """
    def __init__(self, center, start_point, end_point):
        """ """
        self.x0, self.y0 = center
        x1, y1 = start_point
        x2, y2 = end_point
        self.r = np.sqrt((x1-self.x0)**2 + (y1-self.y0)**2)
        assert np.abs(self.r - (np.sqrt((x2-self.x0)**2 + (y2-self.y0)**2))) < 10e-13, \
            'center is not at proper place'
        self.start_theta = angle(center, start_point)
        self.end_theta = angle(center, end_point)
        if self.end_theta < self.start_theta:
            self.end_theta += 2 * np.pi

    # o in [0, 1]
    def gamma(self, o):
        theta = o * (self.end_theta - self.start_theta) + self.start_theta
        return self.x0 + self.r * np.cos(theta), self.y0 + self.r * np.sin(theta)

    def dgamma(self, o):
        theta = o * (self.end_theta - self.start_theta) + self.start_theta
        return -self.r * np.sin(theta) * (self.end_theta - self.start_theta), \
            self.r * np.cos(theta) * (self.end_theta - self.start_theta)
