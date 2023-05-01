# -*- coding: utf-8 -*-
"""The numerical quadrature class.

@author: Lorenzo & Yi
Created on Sat Sep  1 23:58:01 2018
Aerodynamics, AE
TU Delft
"""
import numpy as np
from functools import partial
from scipy.special import legendre, roots_legendre


class Quadrature(object):
    """ Here we store the class for 1d-quadrature nodes and weights. """
    def __init__(self, p, category='Gauss'):
        """
        Initialize.
        
        Parameters
        ----------
        p : int, list or tuple
            The polynomials order. When ndim=1, p is an int; when ndim=2, p is a tuple
            of two int; when ndim=3, p is a tuple of three int.
        category : optional
            The polynomials type.
            
        """
        # check ndim, p, category
        p = (p,) if isinstance(p, (int, float)) else p
        self._ndim_ = len(p)
        assert all([pi >= 0 and pi % 1 == 0 for pi in p]), " <Quadrature> : p = {} is wrong.".format(p)
        self._p_ = p
        _category_ = [category for _ in range(self.ndim)] if isinstance(category, str) else category
        assert all([ci in self.___PRIVATE_coded_quadrature___() for ci in _category_]), \
            f"quad category = {_category_} wrong."
        self._category_ = _category_
        self.___PRIVATE_check_p___()
        self._quad_ = None

    @classmethod
    def ___PRIVATE_coded_quadrature___(cls):
        """ 
        Notice that here the 'Chebyshev' quadrature can not be used in the same way as
        'Gauss' or 'Lobatto' quadratures.
        
        For 'Gauss' or 'Lobatto', we use it to integrate f(x) as :

        \\int{f(x)} = \\sum w_i * f(x_i).
        
        For 'Chebyshev', we do not use it to integrate f(x), but use it to integrate
        f(x)/sqrt(1-x^2). So:
            `\\int{f(x)/sqrt{1-x^2}} = \\sum w_i * f(x_i)`
        This is the first case of the 'Chebyshev' quadrature. 
        
        There is a second case which is to integrate sqrt{1-x^2}*g(x) as 
            \\int{sqrt{1-x^2}*g(x)} = \\sum w_i * g(x_i),
        where these w_i are different. These weights are not included in this class.
        
        As for 'extended_Gauss' quadrature, just do NOT use it. I am not sure if it is
        right here.
        
        """
        return 'Gauss', 'Lobatto'
    
    def ___PRIVATE_check_p___(self):
        """ """
        for i, ci in enumerate(self.category):
            if ci == 'Gauss':
                pass
            elif ci == 'Lobatto':
                assert self.p[i] >= 1
            else:
                raise Exception()

    @property
    def ndim(self):
        """(int) Dimensions."""
        return self._ndim_
    
    @property
    def p(self):
        """(Tuple[int]) Quadrature order p. """
        return self._p_
    
    @property
    def category(self):
        """(List[str], str)"""
        return self._category_

    @property
    def quad(self):
        """(Tuple) ``quad[0]`` are the nodes, ``quad[1]`` are the weights."""
        if self._quad_ is None:
            if self.ndim == 1:
                self._quad_ = getattr(self, '___PRIVATE_compute_'+self.category[0]+'___')(self.p[0])
            else:
                self._quad_ = ([], [])
                for i in range(self.ndim):
                    nodes, weights = getattr(self, '___PRIVATE_compute_'+self.category[i]+'___')(self.p[i])
                    self._quad_[0].append(nodes)
                    self._quad_[1].append(weights)
        return self._quad_
        
    @property
    def quad_ndim(self):
        """ 
        The same as the ``quad`` but now we have tensor product them; so we have high dimensional quad
        nodes and the nd weights; in total the quad_ndim is of shape(n+1,x,y,z)
        """
        if self.ndim == 1:
            quad_ndim = self.quad
        else:
            temp_weights = self.quad[1][0]
            for i in range(self.ndim):
                if i == 0:
                    pass
                else:
                    temp_weights = np.tensordot(temp_weights, self.quad[1][i], axes=0)
            nodes = np.meshgrid(*self.quad[0], indexing='ij')
            quad_ndim = *nodes, temp_weights
        return quad_ndim
        
    @property
    def quad_ndim_ravel(self):
        """Same as `quad_ndim` but now we have raveled it, so it is of shape (n+1, x*y*z)."""
        return [qn.ravel('F') for qn in self.quad_ndim]

    def ___PRIVATE_compute_Lobatto___(self, p):
        """ """
        x_0 = np.cos(np.arange(1, p) / p * np.pi)
        nodal_pts = np.zeros((p + 1))
        # final and initial pt ...
        nodal_pts[0] = 1
        nodal_pts[-1] = -1
        # Newton method for root finding ...
        for i, ch_pt in enumerate(x_0):
            leg_p = partial(self.___PRIVATE_legendre_prime_lobatto___, n=p)
            leg_pp = partial(self.___PRIVATE_legendre_double_prime___, n=p)
            nodal_pts[i + 1] = self.___PRIVATE_newton_method___(leg_p, leg_pp, ch_pt, 100)
        # weights ...
        weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts))**2)
        return nodal_pts[::-1], weights

    @staticmethod
    def ___PRIVATE_compute_Gauss___(p):
        """
        Gauss quadrature are most wildly used, so we cache it.
        """
        # return np.polynomial.legendre.leggauss(p+1)
        # return _gg(p)
        return roots_legendre(p+1)

    @staticmethod
    def ___PRIVATE_legendre_prime___(x, n):
        """
        Calculate first derivative of the nth Legendre Polynomial recursively.
        
        Parameters
        ----------
        x :
            (float,np.array) = domain.
        n :
            (int) = degree of Legendre polynomial (L_n).
            
        Returns
        -------
        legendre_p :
            (np.array) = value first derivative of L_n.
            
        """
        # P'_n+1 = (2n+1) P_n + P'_n-1
        # where P'_0 = 0 and P'_1 = 1
        # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
        if n == 0:
            if isinstance(x, np.ndarray):
                return np.zeros(len(x))
            elif isinstance(x, (int, float)):
                return 0
        if n == 1:
            if isinstance(x, np.ndarray):
                return np.ones(len(x))
            elif isinstance(x, (int, float)):
                return 1
        legendre_p = (n * legendre(n - 1)(x) - n * x * legendre(n)(x))/(1-x**2)
        return legendre_p
    
    def ___PRIVATE_legendre_prime_lobatto___(self, x, n):
        return (1-x**2)**2*self.___PRIVATE_legendre_prime___(x, n)
    
    def ___PRIVATE_legendre_double_prime___(self, x, n):
        """
        Calculate second derivative legendre polynomial recursively.
    
        Parameters
        ----------
        x :
            (float,np.array) = domain.
        n :
            (int) = degree of Legendre polynomial (L_n).
            
        Returns
        -------
        legendre_pp :
            (np.array) = value second derivative of L_n.
            
        """
        legendre_pp = 2 * x * self.___PRIVATE_legendre_prime___(x, n) - n * (n + 1) * legendre(n)(x)
        return legendre_pp * (1 - x ** 2)

    @staticmethod
    def ___PRIVATE_newton_method___(f, df_dx, x_0, n_max, min_error=np.finfo(float).eps * 10):
        """
        Newton method for root finding.
    
        It makes sure quadratic convergence given f'(root) != 0 and abs(f'(ξ)) < 1 over
        the domain considered.
    
        Parameters
        ----------
        f :
            (obj func) = function
        df_dx :
            (obj func) = derivative of f
        x_0 :
            (float) = starting point
        n_max :
            (int) = max number of iterations
        min_error :
            (float) = min allowed error
    
        Returns
        -------
        x[-1] :
            (float) = root of f
            x (np.array) = history of convergence
            
        """
        x = [x_0, ]
        for i in range(n_max - 1):
            x.append(x[i] - f(x[i]) / df_dx(x[i]))
            if abs(x[i + 1] - x[i]) < min_error:
                return x[-1]
        print('WARNING : Newton did not converge to machine precision \nRelative error : ',
              x[-1] - x[-2])
        return x[-1]


if __name__ == '__main__':
    # python ./tools/quadrature.py
    # a = np.polynomial.legendre.leggauss(2)

    a = Quadrature((2, 2, 2), 'Lobatto')
    _ = a.quad_ndim_ravel

    print(_)
