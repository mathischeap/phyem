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
matplotlib.use('TkAgg')


class MsePyMeshVisualizeMatplot(Frozen):
    """"""

    def __init__(self, mesh):
        self._mesh = mesh
        self._freeze()

    def __call__(
        self,
        sampling_factor=1,
        figsize=(10, 6),
        aspect='equal',
        usetex=True,
        labelsize=12,
        ticksize=12,
        xlim=None, ylim=None,
        saveto=None,
        linewidth=0.75,
        color='k',
        title=True,  # True, False, None or custom

    ):
        """Default matplot method."""

        mesh_data_Lines = self._mesh.visualize._generate_mesh_grid_data(sampling_factor=sampling_factor)
        plt.rc('text', usetex=usetex)

        ndim = self._mesh.ndim
        esd = self._mesh.esd

        if esd in (1, 2):  # we use 2-d plot.
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect(aspect)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            plt.xlabel(r"$x$", fontsize=labelsize)
            plt.ylabel(r"$y$", fontsize=labelsize)
            plt.tick_params(axis='both', which='both', labelsize=ticksize)

            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

            if ndim == esd == 1:
                for i in mesh_data_Lines:  # region # i
                    lines = mesh_data_Lines[i]
                    for j, axis_lines in enumerate(lines):

                        _1d_segments = axis_lines[0].transpose()
                        for segment in _1d_segments:
                            plt.plot(segment, [0, 0], linewidth=linewidth, color=color)
                            plt.scatter(segment, [0, 0], color='k')

            elif ndim == esd == 2:
                for i in mesh_data_Lines:  # region # i
                    lines = mesh_data_Lines[i]
                    for j, axis_lines in enumerate(lines):
                        axis0, axis1 = axis_lines
                        if j == 0:
                            plt.plot(axis0, axis1, linewidth=linewidth, color=color)
                        elif j == 1:
                            plt.plot(axis0.T, axis1.T, linewidth=linewidth, color=color)
                        else:
                            raise Exception
            else:
                raise NotImplementedError(f"not implemented for {ndim}-d mesh in {esd}-d space.")

        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.tick_params(labelsize=ticksize)
            ax.set_xlabel(r'$x$', fontsize=labelsize)
            ax.set_ylabel(r'$y$', fontsize=labelsize)
            ax.set_zlabel(r'$z$', fontsize=labelsize)
            x_lim, y_lim, z_lim = [list() for _ in range(3)]

            if ndim == 3:  # plot a 3d mesh in a 3d space.
                for i in mesh_data_Lines:  # region # i
                    lines = mesh_data_Lines[i]
                    for j, axis_lines in enumerate(lines):
                        axis0, axis1, axis2 = axis_lines
                        if aspect == 'equal':
                            x_lim.extend([np.min(axis0), np.max(axis0)])
                            y_lim.extend([np.min(axis1), np.max(axis1)])
                            z_lim.extend([np.min(axis2), np.max(axis2)])
                        else:
                            pass
                        if j == 0:
                            S1, S2 = np.shape(axis0)[1:]
                            for m in range(S1):
                                for n in range(S2):
                                    plt.plot(
                                        axis0[:, m, n], axis1[:, m, n], axis2[:, m, n],
                                        color=color, linewidth=linewidth,
                                    )
                        elif j == 1:
                            S0, S2 = np.shape(axis0)[0], np.shape(axis0)[2]
                            for m in range(S0):
                                for n in range(S2):
                                    plt.plot(
                                        axis0[m, :, n], axis1[m, :, n], axis2[m, :, n],
                                        color=color, linewidth=linewidth,
                                    )
                        elif j == 2:
                            S0, S1 = np.shape(axis0)[:2]
                            for m in range(S0):
                                for n in range(S1):
                                    plt.plot(
                                        axis0[m, n, :], axis1[m, n, :], axis2[m, n, :],
                                        color=color, linewidth=linewidth,
                                    )
                        else:
                            raise Exception()

            else:
                raise NotImplementedError()

            if aspect == 'equal':
                x_lim.sort()
                y_lim.sort()
                z_lim.sort()
                x_lim = [x_lim[0], x_lim[-1]]
                y_lim = [y_lim[0], y_lim[-1]]
                z_lim = [z_lim[0], z_lim[-1]]
                ax.set_box_aspect((np.ptp(x_lim), np.ptp(y_lim), np.ptp(z_lim)))
            else:
                pass

        # deal with title -----------------------------------------------
        if title is True:  # use the default title
            sym_repr = self._mesh.abstract._sym_repr
            plt.title(rf"${sym_repr}$")
        elif title is None or title is False:
            pass
        else:
            plt.title(title)

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='MsePyMeshVisualization')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
        return fig
