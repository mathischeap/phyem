# -*- coding: utf-8 -*-
r"""
These two-dimensional tools are currently working for three-dimensional cases. For
example, for a three-dimensional meshComponents, if on the third-dimension, it is just a
transformation mapping, then we can use a two-dimensional transfinite mapping to get 
the mapping.

They will be deprecated when the 2D transfinite interpolation and the edge functions
are officially coded. But it looks like this will never be happening.

Yi Zhang (C)
Created on Mon Feb 25 15:56:49 2019
Aerodynamics, AE
TU Delft
"""
import numpy as np
from tools.frozen import Frozen
from tools.numerical.space._2d.Jacobian_21 import NumericalJacobianXYt21


class TransfiniteMapping(Frozen):
    """"""
    def __init__(self, gamma, d_gamma):
        """
         y          - 2 +
         ^     _______________
         |    |       R       |
         |    |               | 
         |  + |               | +
         |  3 | U           D | 1
         |  - |               | -
         |    |       L       | 
         |    |_______________|
         |          - 0 +
         |_______________________> x
        
        The indices of `gamma` and `dgamma` are as above. And the directions of the
        mappings are indicated as well.
        
        Parameters
        ----------
        gamma : 
            A tuple of the four boundary functions
            gamma = (
                L(r),
                D(s),
                R(r),
                U(s),
            )
        d_gamma :
            A tuple of first derivatives of gamma.
            d_gamma = (
                [dx/dr, dy/dr], # for L(r), one function return two outputs
                [dx/ds, dy/ds], # for D(s), one function return two outputs
                [dx/dr, dy/dr], # for R(r), one function return two outputs
                [dx/ds, dy/ds], # for U(s), one function return two outputs
            )
            
        """
        t = np.linspace(0, 1, 12)[1:-1]
        _dict_ = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
        for i in range(4):
            XY = gamma[i]
            XtYt = d_gamma[i]
            NJ21 = NumericalJacobianXYt21(XY)
            assert all(NJ21.check_Jacobian(XtYt, t)), \
                " <TransfiniteMapping> :  '{}' edge mapping or Jacobian wrong.".format(_dict_[i])
            
        self.gamma = gamma
        self.d_gamma = d_gamma
        self.gamma1_x0, self.gamma1_y0 = self.gamma[0](0.0)
        self.gamma1_x1, self.gamma1_y1 = self.gamma[0](1.0)
        self.gamma3_x0, self.gamma3_y0 = self.gamma[2](0.0)
        self.gamma3_x1, self.gamma3_y1 = self.gamma[2](1.0)
        self._freeze()
    
    def mapping(self, r, s):
        """
        mapping (r, s) = (0, 1)^2 into (x, y) using the transfinite mapping.
        
        """
        gamma1_xs, gamma1_ys = self.gamma[0](r)
        gamma2_xt, gamma2_yt = self.gamma[1](s)
        gamma3_xs, gamma3_ys = self.gamma[2](r)
        gamma4_xt, gamma4_yt = self.gamma[3](s)
        x = (1-r)*gamma4_xt + r*gamma2_xt + (1-s)*gamma1_xs + s*gamma3_xs - \
            (1-r)*((1-s)*self.gamma1_x0 + s*self.gamma3_x0) - r*((1-s)*self.gamma1_x1 + s*self.gamma3_x1)
        y = (1-r)*gamma4_yt + r*gamma2_yt + (1-s)*gamma1_ys + s*gamma3_ys - \
            (1-r)*((1-s)*self.gamma1_y0 + s*self.gamma3_y0) - r*((1-s)*self.gamma1_y1 + s*self.gamma3_y1)
        return x, y

    def x(self, r, s):
        gamma1_xs = self.gamma[0](r)[0]
        gamma2_xt = self.gamma[1](s)[0]
        gamma3_xs = self.gamma[2](r)[0]
        gamma4_xt = self.gamma[3](s)[0]
        x = (1-r)*gamma4_xt + r*gamma2_xt + (1-s)*gamma1_xs + s*gamma3_xs - \
            (1-r)*((1-s)*self.gamma1_x0 + s*self.gamma3_x0) - r*((1-s)*self.gamma1_x1 + s*self.gamma3_x1)
        return x

    def y(self, r, s):
        gamma1_ys = self.gamma[0](r)[1]
        gamma2_yt = self.gamma[1](s)[1]
        gamma3_ys = self.gamma[2](r)[1]
        gamma4_yt = self.gamma[3](s)[1]
        y = (1-r)*gamma4_yt + r*gamma2_yt + (1-s)*gamma1_ys + s*gamma3_ys - \
            (1-r)*((1-s)*self.gamma1_y0 + s*self.gamma3_y0) - r*((1-s)*self.gamma1_y1 + s*self.gamma3_y1)
        return y

    def dx_dr(self, r, s):
        """ """
        gamma2_xt, gamma2_yt = self.gamma[1](s)
        gamma4_xt, gamma_4yt = self.gamma[3](s)
        dgamma1_xds, dgamma1_yds = self.d_gamma[0](r)
        dgamma3_xds, dgamma3_yds = self.d_gamma[2](r)
        dx_dxi_result = (
                -gamma4_xt + gamma2_xt + (1-s)*dgamma1_xds + s*dgamma3_xds +
                ((1-s)*self.gamma1_x0 + s*self.gamma3_x0) - ((1-s)*self.gamma1_x1 + s*self.gamma3_x1))
        return dx_dxi_result
    
    def dx_ds(self, r, s):
        """ """
        gamma1_xs, gamma1_ys = self.gamma[0](r)
        gamma3_xs, gamma3_ys = self.gamma[2](r)
        dgamma2_xdt, dgamma2_ydt = self.d_gamma[1](s)
        dgamma4_xdt, dgamma4_ydt = self.d_gamma[3](s)
        dx_deta_result = (
                (1-r)*dgamma4_xdt + r*dgamma2_xdt - gamma1_xs + gamma3_xs -
                (1-r)*(-self.gamma1_x0 + self.gamma3_x0) - r*(-self.gamma1_x1 + self.gamma3_x1))
        return dx_deta_result
    
    def dy_dr(self, r, s):
        """ """
        gamma2_xt, gamma2_yt = self.gamma[1](s)
        gamma4_xt, gamma4_yt = self.gamma[3](s)
        dgamma1_xds, dgamma1_yds = self.d_gamma[0](r)
        dgamma3_xds, dgamma3_yds = self.d_gamma[2](r)
        dy_dxi_result = (
                -gamma4_yt + gamma2_yt + (1-s)*dgamma1_yds + s*dgamma3_yds +
                ((1-s)*self.gamma1_y0 + s*self.gamma3_y0) - ((1-s)*self.gamma1_y1 + s*self.gamma3_y1)) 
        return dy_dxi_result
    
    def dy_ds(self, r, s):
        """ """
        gamma1_xs, gamma1_ys = self.gamma[0](r)
        gamma3_xs, gamma3_ys = self.gamma[2](r)
        dgamma2_xdt, dgamma2_ydt = self.d_gamma[1](s)
        dgamma4_xdt, dgamma4_ydt = self.d_gamma[3](s)
        dy_deta_result = (
                (1-r)*dgamma4_ydt + r*dgamma2_ydt - gamma1_ys + gamma3_ys -
                (1-r)*(-self.gamma1_y0 + self.gamma3_y0) - r*(-self.gamma1_y1 + self.gamma3_y1))
        return dy_deta_result

    def illustrate(self):
        """illustrate self."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect('equal')
        _ = np.linspace(0, 1, 7)
        density = np.linspace(0, 1, 100)
        r_lines = np.meshgrid(_, density, indexing='ij')
        s_lines = np.meshgrid(density, _, indexing='ij')
        r_lines = self.mapping(*r_lines)
        s_lines = self.mapping(*s_lines)
        x, y = r_lines
        for xi, yi in zip(x, y):
            plt.plot(xi, yi, '-k', linewidth=0.8)
        x, y = s_lines
        x = x.T
        y = y.T
        for xi, yi in zip(x, y):
            plt.plot(xi, yi, '-k', linewidth=0.8)
        plt.show()
        plt.close()
