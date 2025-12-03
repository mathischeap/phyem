# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import numpy as np

from phyem.tools.quadrature import Quadrature
from phyem.tools.frozen import Frozen
from phyem.tools.miscellaneous.ndarray_cache import ndarray_key_comparer, add_to_ndarray_cache

_cache_nodes = dict()
_cache_edges = dict()


_cache_1d_polynomials_ = {}


def Lobatto_polynomials_of_degree(p):
    """"""
    if p in _cache_1d_polynomials_:
        return _cache_1d_polynomials_[p]
    else:
        assert isinstance(p, int) and p > 0, f'p={p} wrong, it must be positive int.'
        nodes = Quadrature(p, category='Lobatto').quad[0]
        polynomials = _OneDimPolynomial(nodes)
        _cache_1d_polynomials_[p] = polynomials
        return polynomials


class _OneDimPolynomial(Frozen):
    """
    The 1d polynomial basis function space.
    """
    def __init__(self, nodes):
        """I accept inputs like:
            (1) _1dPolynomial(3) # we will be default use Lobatto nodes!
        """
        assert np.ndim(nodes) == 1, \
            " <Polynomials1D> : nodes={} wrong.".format(nodes)
        assert np.all(np.diff(nodes) > 0) and np.max(nodes) == 1 and np.min(nodes) == -1, \
            " <Polynomials1D> : nodes={} wrong, need to be increasing and bounded in [-1, 1].".format(nodes)
        self._nodes_ = nodes
        str_nodes_key = list()
        for node in nodes:
            if node in (-1, 1):
                str_nodes_key.append(str(node))
            else:
                str_nodes_key.append("%.5f" % node)
        self._str_nodes_key = '|'.join(str_nodes_key)
        self._p_ = np.size(self.nodes) - 1
        self._isKronecker_ = True
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<OneDimPolynomial built on {self._nodes_}" + super_repr

    @property
    def isKronecker(self):
        """"""
        return self._isKronecker_

    @property
    def p(self):
        """(int) Polynomial degree."""
        return self._p_

    @property
    def nodes(self):
        """"""
        return self._nodes_

    @property
    def ndim(self):
        """(int) Return ``1``."""
        return 1

    # must have methods ...
    def node_basis(self, x):
        """Return the lagrange polynomials."""
        return self.lagrange_basis(x)

    def lagrange_basis(self, x):
        """Return the lagrange polynomials evaluated at ``x``."""
        cached, cache_data = ndarray_key_comparer(_cache_nodes, [x, ], check_str=self._str_nodes_key)
        if cached:
            return cache_data
        else:
            pass

        p = np.size(self.nodes)
        basis = np.ones((p, np.size(x)))
        # lagrange basis functions
        for i in range(p):
            for j in range(p):
                if i != j:
                    basis[i, :] *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])

        add_to_ndarray_cache(_cache_nodes, [x, ], basis, check_str=self._str_nodes_key, maximum=10)

        return basis

    def edge_basis(self, x):
        """Return the edge polynomials evaluated at ``x``."""
        cached, cache_data = ndarray_key_comparer(_cache_edges, [x, ], check_str=self._str_nodes_key)
        if cached:
            return cache_data
        else:
            pass
        p = np.size(self.nodes) - 1
        derivatives_poly = self._derivative_poly_(p, x)
        edge_poly = np.zeros((p, np.size(x)))
        for i in range(p):
            for j in range(i + 1):
                edge_poly[i] -= derivatives_poly[j, :]

        add_to_ndarray_cache(_cache_edges, [x, ], edge_poly, check_str=self._str_nodes_key, maximum=10)

        return edge_poly

    @property
    def node_reference_mass_matrix(self):
        """(numpy.ndarray) The mass matrix for the lagrange polynomials."""
        quad_nodes, quad_weights = Quadrature(np.size(self.nodes) + 1).quad
        quad_basis = self.lagrange_basis(x=quad_nodes)
        M = np.einsum(
            'ik,jk,k->ij',
            quad_basis, quad_basis, quad_weights,
            optimize='optimal'
        )
        return M

    @property
    def edge_reference_mass_matrix(self):
        """(numpy.ndarray) The mass matrix for the edge polynomials."""
        quad_nodes, quad_weights = Quadrature(np.size(self.nodes) + 1).quad
        quad_basis = self.edge_basis(x=quad_nodes)
        M = np.einsum(
            'ik,jk,k->ij',
            quad_basis, quad_basis, quad_weights,
            optimize='optimal'
        )
        return M

    @staticmethod
    def __derivative_poly_nodes__(p, nodes):
        r"""
        For computation of the derivative at the nodes a more efficient and
        accurate formula can be used, see [1]:
                 |
                 | frac{c_{k}}{c_{j}}\frac{1}{x_{k}-x_{j}}, k \neq j.
                 |
        d_{kj} = <
                 |
                 | sum_{l=1,l\neq k}^{p+1}\frac{1}{x_{k}-x_{l}}, k = j.
                 |
        with
        c_{k} = prod_{l=1,l\neq k}^{p+1} (x_{k}-x_{l}).

        Parameters
        ----------
        p : int
            degree of polynomial.
        nodes : ndarray
            Lagrange nodes.
            [1] Costa, B., Don, W. S.: On the computation of high order pseudo-spectral
                derivatives, Applied Numerical Mathematics, vol.33 (1-4), pp. 151-159.

        """
        # compute distances between the nodes
        # noinspection PyUnresolvedReferences
        xi_xj = nodes.reshape(p + 1, 1) - nodes.reshape(1, p + 1)
        # diagonals to one
        xi_xj[np.diag_indices(p + 1)] = 1
        # compute (ci's)
        c_i = np.prod(xi_xj, axis=1)
        # compute ci/cj = ci_cj(i,j)
        c_i_div_cj = np.transpose(c_i.reshape(1, p + 1) / c_i.reshape(p + 1, 1))
        # result formula
        derivative = c_i_div_cj / xi_xj
        # put the diagonals equal to zeros
        derivative[np.diag_indices(p + 1)] = 0
        # compute the diagonal values enforcing sum over rows = 0
        derivative[np.diag_indices(p + 1)] = -np.sum(derivative, axis=1)
        return derivative

    def _derivative_poly_(self, p, x):
        """Return the derivatives of the polynomials in the domain x."""
        nodal_derivative = self.__derivative_poly_nodes__(p, self.nodes)
        polynomials = self.lagrange_basis(x)
        return np.transpose(nodal_derivative) @ polynomials

    def plot_lagrange_basis(self, dual=False, plot_density=300, ylim_ratio=0.1,
                            title=True, left=0.15, bottom=0.15,
                            minor_tick_length=5, major_tick_length=10, tick_pad=7,
                            tick_size=15, label_size=15, title_size=15,
                            linewidth=1.2, saveto=None, figsize=(6, 4), usetex=True
                            ):
        """

        Parameters
        ----------
        dual
        plot_density
        ylim_ratio
        title : {str, bool}
        left
        bottom
        minor_tick_length
        major_tick_length
        tick_pad
        tick_size
        label_size
        title_size
        linewidth
        saveto
        figsize
        usetex

        Returns
        -------

        """
        plt.rc('text', usetex=usetex)
        if usetex:
            plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
        x = np.linspace(-1, 1, plot_density)
        basis = self.lagrange_basis(x=x)
        if dual:
            M = self.node_reference_mass_matrix
            M = np.linalg.inv(M)
            basis = np.einsum('ik,ij->jk', basis, M, optimize='optimal')
        bmx = np.max(basis)
        bmi = np.min(basis)
        interval = bmx - bmi
        ylim = [bmi - interval * ylim_ratio, bmx + interval * ylim_ratio]
        fig = plt.figure(figsize=figsize)
        for basis_i in basis:
            plt.plot(x, basis_i, linewidth=1 * linewidth)
        for i in self.nodes:
            plt.plot([i, i], ylim, '--', color=(0.2, 0.2, 0.2, 0.2), linewidth=0.8 * linewidth)
        if not dual:
            plt.plot([-1, 1], [1, 1], '--', color=(0.2, 0.2, 0.2, 0.2), linewidth=0.8 * linewidth)
        plt.plot([-1, 1], [0, 0], '--', color=(0.5, 0.5, 0.5, 1), linewidth=0.8 * linewidth)
        if title is True:
            if dual:
                title = 'dual Lagrange polynomials'
            else:
                title = 'Lagrange polynomials'
            plt.title(title, fontsize=title_size)
        elif title is False:
            pass
        elif title is not None:
            plt.title(title, fontsize=title_size)
        else:
            pass
        plt.gcf().subplots_adjust(left=left)
        plt.gcf().subplots_adjust(bottom=bottom)
        plt.ylim(ylim)
        plt.xlim([-1, 1])
        plt.tick_params(which='both', labeltop=False, labelright=False, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=minor_tick_length)
        plt.tick_params(axis='both', which='major', direction='in', length=major_tick_length)
        plt.tick_params(axis='both', which='both', labelsize=tick_size)
        plt.tick_params(axis='x', which='both', pad=tick_pad)
        # plt.legend(fontsize=legend_size, loc=legend_local, frameon=legend_frame)
        plt.xlabel(r"$\lambda$", fontsize=label_size)
        if dual:
            plt.ylabel(r"$\widetilde{l}^{i}(\xi)$", fontsize=label_size)
        else:
            plt.ylabel(r"$l^{i}(\lambda)$", fontsize=label_size)
        if saveto is not None:
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='MsePyMeshVisualization')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])

    def plot_edge_basis(
        self, dual=False, plot_density=300, ylim_ratio=0.1,
        title=True, left=0.15, bottom=0.15, fill_between=1,
        minor_tick_length=5, major_tick_length=10, tick_pad=7,
        tick_size=15, label_size=15, title_size=15,
        linewidth=1.2, saveto=None, figsize=(6, 4), usetex=True,
    ):
        """

        Parameters
        ----------
        dual
        plot_density
        ylim_ratio
        title : {str, bool}
        left
        bottom
        fill_between
        minor_tick_length
        major_tick_length
        tick_pad
        tick_size
        label_size
        title_size
        linewidth
        saveto
        figsize
        usetex

        Returns
        -------

        """
        plt.rc('text', usetex=usetex)
        if usetex:
            plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
        # _____ preparing the x ________________________________________________________
        x = np.linspace(-1, 1, plot_density)
        x = np.concatenate((x, self.nodes))
        x.sort()
        x = list(x)
        segnodes = ()
        for i in range(self.p + 1):
            segnodes += (x.index(self.nodes[i]),)
        # _______ LET GET THE data 2 be plotted _________________________________________
        basis = self.edge_basis(np.array(x))
        if dual:
            M = self.edge_reference_mass_matrix
            M = np.linalg.inv(M)
            basis = np.einsum('ik,ij->jk', basis, M, optimize='optimal')
        bmx = np.max(basis)
        bmi = np.min(basis)
        interval = bmx - bmi
        if interval == 0:
            interval = 0.1
        ylim = [bmi - interval * ylim_ratio, bmx + interval * ylim_ratio]
        # ___________ do THE PLOT ______________________________________________________
        fig = plt.figure(figsize=figsize)
        for basis_i in basis:
            plt.plot(x, basis_i, linewidth=1 * linewidth)
        for i in self.nodes:
            plt.plot([i, i], ylim, '--', color=(0.2, 0.2, 0.2, 0.2), linewidth=0.8 * linewidth)
        plt.plot([-1, 1], [0, 0], '--', color=(0.5, 0.5, 0.5, 1), linewidth=0.8 * linewidth)
        # _____ Titling ________________________________________________________________
        if title is True:
            if dual:
                title = 'dual edge polynomials'
            else:
                title = 'edge polynomials'
            plt.title(title, fontsize=title_size)
        elif title is False:
            pass
        elif title is not None:
            plt.title(title, fontsize=title_size)
        else:
            pass
        # ________________ adjusting and labeling ______________________________________
        plt.gcf().subplots_adjust(left=left, bottom=bottom)
        # plt.gcf().subplots_adjust(bottom=bottom)
        plt.ylim(ylim)
        plt.xlim([-1, 1])

        plt.tick_params(which='both', labeltop=False, labelright=False, top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', length=minor_tick_length)
        plt.tick_params(axis='both', which='major', direction='in', length=major_tick_length)
        plt.tick_params(axis='both', which='both', labelsize=tick_size)
        plt.tick_params(axis='x', which='both', pad=tick_pad)
        plt.tick_params(axis='y', which='both', pad=tick_pad)

        # plt.legend(fontsize=legend_size, loc=legend_local, frameon=legend_frame)

        plt.xlabel(r"$\lambda$", fontsize=label_size)
        if dual:
            plt.ylabel(r"$\widetilde{e}^{i}(\lambda)$", fontsize=label_size)
        else:
            plt.ylabel(r"$e^{i}(\lambda)$", fontsize=label_size)
        # _______ fill between ...
        if fill_between is not None and not dual:  # fill_between is on and dual is False
            if isinstance(fill_between, int):
                fill_between = (fill_between, fill_between)
            # ______ INT: fill along one edge, different filling for one segment ...
            if isinstance(fill_between, (tuple, list)):
                assert np.shape(fill_between) == (2,), \
                    ' <Polynomials1D> : fill_between={} wrong.'.format(fill_between)
                efid, segid = fill_between  # start counting at 1
                efid -= 1  # start counting at 0
                segid -= 1  # start counting at 0
                for i in range(self.p):
                    xi = x[segnodes[i]:segnodes[i + 1] + 1]
                    yi = basis[efid][segnodes[i]:segnodes[i + 1] + 1]
                    if i == segid:  # the highlighted `fill_between[1]`th segment
                        plt.fill_between(xi, 0, yi, color='grey', alpha=0.4)
                    else:
                        plt.fill_between(xi, 0, yi, color='grey', alpha=0.2)
            # ____ ELSE: error .......................................................
            else:
                raise Exception(' <Polynomials1D> : fill_between={} wrong.'.format(fill_between))
            # -------------------------------------------------------------------------------------
        # -------------------------------------------------
        if saveto is not None:
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='MsePyMeshVisualization')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])


if __name__ == "__main__":
    # python msepy/tools/polynomials.py
    nodes = Quadrature(2, category='Lobatto').quad_nodes
    # print(nodes.__class__)
    p1 = _OneDimPolynomial(np.array(nodes))
    p1.plot_lagrange_basis(  # plot_lagrange_basis
        dual=False,
        title=False,
        figsize=(6, 4), tick_size=20, label_size=20,  # fill_between=2,
    )

    # p1.plot_edge_basis(
    #     dual=False,
    #     title=False,
    #     figsize=(6, 4), tick_size=20, label_size=20,  fill_between=1,
    # )
