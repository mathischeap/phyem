# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK, COMM


class MseHttGreatMeshVisualize_TS_Hierarchy(Frozen):
    r""""""
    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        ts = self._tgm.___original_ts___
        if ts == 0:
            return None
        elif ts == 1:
            return self.___ts1_plotter___(*args, **kwargs)
        else:
            raise NotImplementedError()

    def ___ts1_plotter___(self, *args, ddf=1, **kwargs):
        r""""""
        ts1_data_Lines = self._tgm.visualize._generate_element_outline_data(ddf=ddf, internal_grid=0)
        ts1_data_Lines = COMM.gather(ts1_data_Lines, root=MASTER_RANK)

        ts0_data_Lines = self._tgm.ts(0)._generate_element_outline_data(ddf=ddf, internal_grid=0)
        ts0_data_Lines = COMM.gather(ts0_data_Lines, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return None
        else:
            pass

        data = {}
        for _ in ts1_data_Lines:
            data.update(_)
        ts1_data_Lines = data

        data = {}
        for _ in ts0_data_Lines:
            data.update(_)
        ts0_data_Lines = data

        mn = set()
        for i in ts1_data_Lines:
            mn.add(ts1_data_Lines[i]['mn'])
        for i in ts0_data_Lines:
            mn.add(ts0_data_Lines[i]['mn'])

        if len(mn) == 1 and list(mn)[0] == (2, 2):
            return self.___ts1_plotter___m2n2___(ts0_data_Lines, ts1_data_Lines, *args, **kwargs)
        else:
            raise NotImplementedError()

    @classmethod
    def ___ts1_plotter___m2n2___(
            cls,
            ts0_data_Lines, ts1_data_Lines,

            figsize=(8, 6),
            aspect='equal',
            usetex=True,

            labelsize=12,

            ticksize=12,
            xticks=None, yticks=None,
            minor_tick_length=4, major_tick_length=8,

            xlim=None, ylim=None,
            saveto=None,
            linewidth=0.75,
            colors=('k', 'b'),
            title=None,  # None or custom
            data_only=False,

            pad_inches=0
    ):
        r""""""
        assert RANK == MASTER_RANK, f"plot routine only in the master rank."
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
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        ax.tick_params(labelsize=ticksize)
        plt.tick_params(axis='both', which='minor', direction='out', length=minor_tick_length)
        plt.tick_params(axis='both', which='major', direction='out', length=major_tick_length)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        for i in ts1_data_Lines:  # element # i

            lines = ts1_data_Lines[i]

            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                elif isinstance(line_index, str) and line_index[:13] == 'internal_line':
                    raise Exception(f"ts plot no internal lines")
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=0.85*linewidth, color=colors[1])

        for i in ts0_data_Lines:  # element # i

            lines = ts0_data_Lines[i]

            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                elif isinstance(line_index, str) and line_index[:13] == 'internal_line':
                    raise Exception(f"ts plot no internal lines")
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=linewidth, color=colors[0])

        # deal with title -----------------------------------------------
        if title is None:
            title = r"the great mesh"
            plt.title(title)
        elif title is False:
            pass
        else:
            plt.title(title)

        # ---------------------------------------------------------------
        if data_only:
            return fig
        else:
            # save -----------------------------------------------------------
            if saveto is not None and saveto != '':
                plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
            else:
                from src.config import _setting, _pr_cache
                if _setting['pr_cache']:
                    _pr_cache(fig, filename='msehtt_elements')
                else:
                    plt.tight_layout()
                    plt.show(block=_setting['block'])
            plt.close()
            return None
