# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.matplot.plot import plot
from tools.matplot.contour import contour, contourf
from tools.matplot.quiver import quiver


class MsePyRootFormVisualizeMatplot(Frozen):
    """"""
    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, title=None, **kwargs):
        """Call the default plotter coded in this module as well."""
        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k

        if title is None:
            title = r'$t=%.3f$' % self._f.visualize._t  # this visualize is for cochain @ t
        elif title is False:
            title = None
        else:
            pass

        indicator = self._f._space.abstract.indicator

        if indicator in ('Lambda', 'bundle-diagonal'):
            return getattr(self, f'_Lambda_m{m}_n{n}_k{k}')(title=title, **kwargs)

        elif indicator == 'bundle':
            if k in (0, n):  # vector forms
                return getattr(self, f'_Lambda_m{m}_n{n}_k1')(**kwargs)
            else:  # tensor forms
                return getattr(self, f'_bundle_m{m}_n{n}_k1')(**kwargs)

        else:
            raise NotImplementedError(f"msepy matplot not implemented for {indicator}")

    def _Lambda_m1_n1_k0(
            self, sampling_factor=1,
            figsize=(10, 6),
            color='k',
            **kwargs
    ):
        """"""
        samples = 500 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 100:
            samples = 100
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        x, v = self._f[t].reconstruct(linspace)

        x = x[0]
        v = v[0]
        num_lines = len(x)  # also num elements
        fig = plot(
            x, v, num_lines=num_lines, xlabel='$x$', labels=False, styles='-',
            figsize=figsize, colors=color,
            **kwargs
        )
        return fig

    def _Lambda_m1_n1_k1(self, *args, **kwargs):
        """"""
        return self._Lambda_m1_n1_k0(*args, **kwargs)

    def _Lambda_m2_n2_k0(
            self, sampling_factor=1,
            plot_type='contourf',
            **kwargs
    ):
        """"""
        samples = 20000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 75:
            samples = 75
        elif samples < 7:
            samples = 7
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xy, v = self._f[t].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        x, y, v = self._mesh._regionwsie_stack(x, y, v[0])

        if plot_type == 'contourf':
            fig = contourf(x, y, v, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, **kwargs)
        else:
            raise Exception()

        return fig

    def _Lambda_m2_n2_k1(
            self, sampling_factor=1,
            plot_type='contourf',
            saveto=None,
            title=None,
            title_components=None,
            **kwargs
    ):
        """Plot a msepy scalar-valued 1-form on 2d manifold in 2d space.

        Parameters
        ----------
        sampling_factor
        plot_type :
            {
                'contourf', 'contour', 'quiver',
                'norm-contour', 'norm-contourf',
                'magnitude-contour', 'magnitude-contourf',
            }
            each of which means:
                'contourf': contourf plots of two components;
                'contour': contour plots of two components;
                'quiver': a quiver plot of the vector;
                'norm-contour': a contour plot the norm of the vector;
                'norm-contourf': a contourf plot the norm of the vector;
                'magnitude-contour': a contour plot the magnitude of the vector;
                'magnitude-contourf': a contourf plot the magnitude of the vector;

        saveto
        title
        title_components
        kwargs

        Returns
        -------

        """
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))

        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xy, uv = self._f[t].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        u, v = uv
        x, y, u, v = self._mesh._regionwsie_stack(x, y, u, v)
        if plot_type in ('contourf', 'contour'):
            if saveto is None:
                saveto_x = None
                saveto_y = None
            else:
                saveto0, saveto1 = saveto.split('.')
                saveto_x = saveto0 + '_x' + '.' + saveto1
                saveto_y = saveto0 + '_y' + '.' + saveto1

            if title_components is None:
                title_x = '$x$-component'
                title_y = '$y$-component'
            else:
                title_x, title_y = title_components

            if plot_type == 'contourf':
                fig = [
                    contourf(x, y, u, title=title_x, saveto=saveto_x, **kwargs),
                    contourf(x, y, v, title=title_y, saveto=saveto_y, **kwargs)
                ]
            elif plot_type == 'contour':
                fig = [
                    contour(x, y, u, title=title_x, saveto=saveto_x, **kwargs),
                    contour(x, y, v, title=title_y, saveto=saveto_y, **kwargs)
                ]
            else:
                raise Exception()

        elif plot_type == "norm-contour":
            norm = dict()
            for region in x:
                ur = u[region]
                vr = v[region]
                norm[region] = np.sqrt(ur**2 + vr**2)
            fig = contour(x, y, norm, saveto=saveto, title=title, **kwargs)

        elif plot_type == "norm-contourf":
            norm = dict()
            for region in x:
                ur = u[region]
                vr = v[region]
                norm[region] = np.sqrt(ur**2 + vr**2)
            fig = contourf(x, y, norm, saveto=saveto, title=title, **kwargs)

        elif plot_type == "magnitude-contour":
            norm = dict()
            for region in x:
                ur = u[region]
                vr = v[region]
                _ = np.sqrt(ur**2 + vr**2)
                _[_ < 1e-16] = 1e-16
                norm[region] = np.log10(_)
            fig = contour(x, y, norm, saveto=saveto, title=title, **kwargs)

        elif plot_type == "magnitude-contourf":
            norm = dict()
            for region in x:
                ur = u[region]
                vr = v[region]
                _ = np.sqrt(ur**2 + vr**2)
                _[_ < 1e-16] = 1e-16
                norm[region] = np.log10(_)
            fig = contourf(x, y, norm, saveto=saveto, title=title, **kwargs)

        elif plot_type == "quiver":
            fig = self._quiver(x, y, u, v, saveto=saveto, title=title, **kwargs)

        else:
            raise Exception()

        return fig

    def _Lambda_m2_n2_k2(self, **kwargs):
        """"""
        return self._Lambda_m2_n2_k0(**kwargs)

    @staticmethod
    def _quiver(
            x, y, u, v, **kwargs
    ):
        """"""
        X = list()
        Y = list()
        U = list()
        V = list()
        for i in x:
            X.append(x[i])
            Y.append(y[i])
            U.append(u[i])
            V.append(v[i])

        U = np.array(U).ravel()
        V = np.array(V).ravel()
        X = np.array(X).ravel()
        Y = np.array(Y).ravel()
        return quiver(X, Y, U, V, **kwargs)

    def _bundle_m2_n2_k1(
            self, sampling_factor=1,
            plot_type='contourf',
            saveto=None,
            title_components=None,
            **kwargs
    ):
        """"""
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))

        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xy, V = self._f[t].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        V0, V1 = V
        v00, v01 = V0
        v10, v11 = V1
        x, y, v00, v01, v10, v11 = self._mesh._regionwsie_stack(x, y, v00, v01, v10, v11)
        if plot_type in ('contourf', 'contour'):
            if saveto is None:
                saveto_00 = None
                saveto_01 = None
                saveto_10 = None
                saveto_11 = None
            else:
                saveto0, saveto1 = saveto.split('.')
                saveto_00 = saveto0 + '_00' + '.' + saveto1
                saveto_01 = saveto0 + '_01' + '.' + saveto1
                saveto_10 = saveto0 + '_10' + '.' + saveto1
                saveto_11 = saveto0 + '_11' + '.' + saveto1

            if title_components is None:
                title_00 = '$00$-component'
                title_01 = '$01$-component'
                title_10 = '$10$-component'
                title_11 = '$11$-component'
            else:
                title_00, title_01, title_10, title_11 = title_components

            if plot_type in ('contourf', 'contour'):
                if plot_type == 'contourf':
                    plotter = contourf
                else:
                    plotter = contour

                fig = [
                    plotter(x, y, v00, title=title_00, saveto=saveto_00, **kwargs),
                    plotter(x, y, v01, title=title_01, saveto=saveto_01, **kwargs),
                    plotter(x, y, v10, title=title_10, saveto=saveto_10, **kwargs),
                    plotter(x, y, v11, title=title_11, saveto=saveto_11, **kwargs),
                ]

            else:
                raise Exception()

        else:
            raise NotImplementedError(f'Not implemented for plot_type={plot_type}')

        return fig
