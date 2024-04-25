# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')


class MseHttGreatMeshVisualizeMatplot(Frozen):
    """"""

    def __init__(self, tgm):
        """"""
        self._tgm = tgm
        self._freeze()

    def __call__(
        self,
        ddf=1,
        **kwargs
    ):
        """

        Parameters
        ----------
        ddf

        """

        mesh_data_Lines = self._tgm.visualize._generate_element_outline_data(ddf=ddf)
        mesh_data_Lines = COMM.gather(mesh_data_Lines, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        # from now on, we must be in the master rank.

        data = {}
        for _ in mesh_data_Lines:
            data.update(_)

        mn = set()
        for i in data:
            mn.add(data[i]['mn'])

        if len(mn) == 1 and list(mn)[0] == (2, 2):  # all elements are 2d elements in 2d spaces.
            return self._plot_2d_great_mesh_in_2d_space(data, **kwargs)
        else:
            raise NotImplementedError()

    @classmethod
    def _plot_2d_great_mesh_in_2d_space(
            cls,
            line_data,
            figsize=(10, 6),
            aspect='equal',
            usetex=True,
            labelsize=12,
            ticksize=12,
            xlim=None, ylim=None,
            saveto=None,
            linewidth=0.75,
            color='k',
            title=None,  # None or custom
            data_only=False,
    ):
        """

        Parameters
        ----------
        line_data
        figsize
        aspect
        usetex
        labelsize
        ticksize
        xlim
        ylim
        saveto
        linewidth
        color
        title : {str, bool, None}, optional
        data_only

        Returns
        -------

        """
        plt.rc('text', usetex=usetex)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(aspect)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.xlabel(r"$x$", fontsize=labelsize)
        plt.ylabel(r"$y$", fontsize=labelsize)
        plt.tick_params(axis='both', which='both', labelsize=ticksize)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        for i in line_data:  # region # i
            lines = line_data[i]
            for line_index in lines:
                if line_index == 'mn':
                    pass
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=linewidth, color=color)

        # deal with title -----------------------------------------------
        if title is None:
            title = r"the great mesh"
            plt.title(title)
        elif title is False:
            pass
        else:
            plt.title(title)

        if data_only:
            return fig
        else:
            pass

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
