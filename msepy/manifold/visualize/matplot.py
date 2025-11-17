# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK


class MsePyManifoldVisualizeMatplot(Frozen):
    """"""

    def __init__(self, manifold):
        """"""
        self._manifold = manifold
        self._freeze()

    def __call__(
            self,
            refining_factor=1,
            figsize=(10, 6),
            aspect='equal',
            usetex=True,
            labelsize=12,
            ticksize=12,
            xlim=None, ylim=None,
            saveto=None,
            linewidth=0.75,
            color='k',

    ):
        """"""
        if RANK != MASTER_RANK:
            return
        else:
            pass
        manifold_data_lines = self._manifold.visualize._generate_manifold_grid_data(refining_factor=refining_factor)
        plt.rc('text', usetex=usetex)

        ndim = self._manifold.ndim  # aka n also
        esd = self._manifold.esd    # aka m also

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

            if esd == 1:
                if ndim == 1:
                    for i in manifold_data_lines:  # region # i
                        lines = manifold_data_lines[i][0][0]
                        plt.plot(lines, [0, 0], linewidth=linewidth, color=color)
                        plt.scatter(lines, [0, 0], color='k')
                elif ndim == 0:
                    for i in manifold_data_lines:  # region # i
                        nodes = manifold_data_lines[i][0]
                        plt.scatter(nodes, [0], color='k')
                else:
                    raise Exception(f"cannot be!")

            elif esd == 2:
                for i in manifold_data_lines:  # region # i
                    lines = manifold_data_lines[i]
                    for j, line in enumerate(lines):
                        axis0, axis1 = line
                        plt.plot(axis0, axis1, linewidth=linewidth, color=color)
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
                for i in manifold_data_lines:  # region # i
                    lines = manifold_data_lines[i]
                    for line in lines:
                        axis0, axis1, axis2 = line
                        if aspect == 'equal':
                            x_lim.extend([np.min(axis0), np.max(axis0)])
                            y_lim.extend([np.min(axis1), np.max(axis1)])
                            z_lim.extend([np.min(axis2), np.max(axis2)])
                        else:
                            pass
                        plt.plot(
                            axis0, axis1, axis2,
                            color=color, linewidth=linewidth
                        )

            elif ndim == 2:  # plot 2d faces in a 3d space.
                for i in manifold_data_lines:  # region # i
                    face = manifold_data_lines[i]
                    X, Y, Z = face
                    ax.plot_surface(X, Y, Z, color='lightgray')

                    if aspect == 'equal':
                        x_lim.extend([np.min(X), np.max(X)])
                        y_lim.extend([np.min(Y), np.max(Y)])
                        z_lim.extend([np.min(Z), np.max(Z)])
                    else:
                        pass

            else:
                raise NotImplementedError(f"not implemented for {ndim}-d mesh in {esd}-d space.")

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

        plt.tight_layout()
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting
            plt.show(block=_setting['block'])

        return fig


if __name__ == '__main__':
    # python msepy/manifold/visualize/matplot.py
    import __init__ as ph
    space_dim = 3
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    # msepy.config(mnf)('crazy', c=0., periodic=True, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(mnf)('backward_step')
    msepy.config(msh)([3 for _ in range(space_dim)])

    mnf.visualize()
