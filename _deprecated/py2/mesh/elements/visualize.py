# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
from matplotlib import cm


class MseHyPy2MeshElementsVisualize(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def _make_base_mesh(self, sampling_factor=1, top_right_bounds=False):
        """

        Parameters
        ----------
        sampling_factor
        top_right_bounds

        Returns
        -------

        """
        base_mesh_data_Lines = self._elements.background.visualize._generate_mesh_grid_data(
            sampling_factor=sampling_factor
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        if top_right_bounds:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.xlabel(r"$x$", fontsize=15)
        plt.ylabel(r"$y$", fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)

        axis0 = [-1, 1]
        for i in base_mesh_data_Lines:  # region # i
            lines = base_mesh_data_Lines[i]
            for j, axis_lines in enumerate(lines):
                axis0, axis1 = axis_lines
                if j == 0:
                    plt.plot(axis0, axis1, linewidth=0.5, color='lightgray')
                elif j == 1:
                    plt.plot(axis0.T, axis1.T, linewidth=0.5, color='lightgray')
                else:
                    raise Exception
        density = len(axis0)
        return fig, density

    def _make_mesh_data(self, sampling_factor=1):
        """"""
        density = int(5 * sampling_factor)
        if density < 5:
            density = 5
        else:
            pass
        ones_ = np.ones(density)
        space = np.linspace(-1, 1, density)

        edge_00 = (-ones_, space)
        edge_01 = (ones_, space)
        edge_10 = (space, -ones_)
        edge_11 = (space, ones_)

        edge_b = edge_01
        edge_0 = edge_10
        edge_1 = edge_11

        data_dict = dict()
        for i in self._elements:
            fc = self._elements[i]
            if fc._type == 'q':
                data_dict[i] = (
                    fc.ct.mapping(*edge_00),
                    fc.ct.mapping(*edge_01),
                    fc.ct.mapping(*edge_10),
                    fc.ct.mapping(*edge_11),
                )
            elif fc._type == 't':
                data_dict[i] = (
                    fc.ct.mapping(*edge_b),
                    fc.ct.mapping(*edge_0),
                    fc.ct.mapping(*edge_1),
                )
            else:
                raise Exception

        return data_dict

    def __call__(
            self, title=None,
            sampling_factor=1, saveto=None, dpi=200,
            color='gray',
            show_refining_strength_distribution=True, colormap='bwr',
            num_levels=20, levels=None,
            intermediate=False, check=True, top_right_bounds=False,
    ):
        """

        Parameters
        ----------
        title
        sampling_factor
        saveto
        dpi
        color
        show_refining_strength_distribution
        colormap
        num_levels
        levels
        intermediate
        check
        top_right_bounds

        Returns
        -------

        """
        fig, density = self._make_base_mesh(sampling_factor=sampling_factor, top_right_bounds=top_right_bounds)

        if density < 20:
            density = 20
        else:
            pass

        plt.rcParams['image.cmap'] = colormap
        if show_refining_strength_distribution and self._elements.num_levels > 0:
            from tools.matplot.contour import ___set_contour_levels___
            func = self._elements._refining_function
            r = s = np.linspace(0, 1, density*3)
            r, s = np.meshgrid(r, s, indexing='ij')
            X = dict()
            Y = dict()
            v = dict()
            for region in func:
                R = self._elements.background.manifold.regions[region]
                x, y = R._ct.mapping(r, s)
                X[region] = x
                Y[region] = y
                func_region = func[region]
                strength = func_region(x, y)
                v[region] = strength

            if levels is None:
                levels = ___set_contour_levels___(v, num_levels=num_levels)
            else:
                pass

            level_max = np.max(levels)
            level_min = np.min(levels)

            for region in v:
                v_region = v[region]
                v_region[v_region > level_max] = level_max
                v_region[v_region < level_min] = level_min
                plt.contourf(X[region], Y[region], v_region, levels=levels)

            mappable = cm.ScalarMappable()
            mappable.set_array(np.array(levels))
            cb = plt.colorbar(mappable, extend='both')
            cb.ax.tick_params(labelsize=12)

        for level in self._elements._levels:
            fig = level._visualize(fig, density, color=color)

        if title is None:
            sym_repr = self._elements.background.abstract._sym_repr
            plt.title(rf"${sym_repr}$")
        else:
            plt.title(title)

        if check:
            xy = self._elements.ct.mapping(
                np.array([0]), np.array([0]), fc_range=None
            )
            fc_center_dict = dict()
            for fc_index in xy:
                x, y = xy[fc_index]
                x = x[0]
                y = y[0]
                x = round(x, 5)
                y = round(y, 5)
                str_ = f"{x}-{y}"
                assert str_ not in fc_center_dict
                fc_center_dict[str_] = fc_index

        # save -----------------------------------------------------------
        if intermediate:
            return fig
        else:
            if saveto is not None and saveto != '':
                plt.savefig(saveto, bbox_inches='tight', dpi=dpi)
            else:
                from src.config import _setting, _pr_cache

                if _setting['pr_cache']:
                    _pr_cache(fig, filename='msehy_py2_mesh')
                else:
                    matplotlib.use('TkAgg')
                    plt.tight_layout()
                    plt.show(block=_setting['block'])
            plt.close()
            return fig