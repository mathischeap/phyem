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


class MseHtt_MultiGrid_GreatMesh_Visualize(Frozen):
    r""""""

    def __init__(self, mg_gm):
        r""""""
        self._mg_gm = mg_gm
        self._freeze()

    def __call__(self, ddf=1, **kwargs):
        r""""""
        mesh = self._mg_gm

        if RANK == MASTER_RANK:
            lvl_gm_line_data = {}
        else:
            pass

        for lvl in mesh.level_range:
            lvl_gm = mesh.get_level(lvl)
            mesh_data_Lines = lvl_gm.visualize._generate_element_outline_data(ddf=ddf)
            mesh_data_Lines = COMM.gather(mesh_data_Lines, root=MASTER_RANK)

            if RANK == MASTER_RANK:
                data = {}
                for _ in mesh_data_Lines:
                    data.update(_)
                # noinspection PyUnboundLocalVariable
                lvl_gm_line_data[lvl] = data
            else:
                pass

        if RANK != MASTER_RANK:
            return None
        else:
            pass

        mn = set()
        for lvl in lvl_gm_line_data:
            data = lvl_gm_line_data[lvl]
            for i in data:
                mn.add(data[i]['mn'])
        if len(mn) == 1 and list(mn)[0] == (2, 2):  # all elements are 2d elements in 2d spaces.

            return _plot_2d_MG_great_mesh_for_m2n2(
                lvl_gm_line_data,
                **kwargs
            )

        else:
            raise NotImplementedError()


def _plot_2d_MG_great_mesh_for_m2n2(
        lvl_gm_line_data,

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
        colormap='tab10',
        title=None,  # None or custom
        data_only=False,

        pad_inches=0
):
    r""""""
    assert RANK == MASTER_RANK, f"PLS CALL ME FROM MASTER-RANK ONLY."
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

    colors = matplotlib.colormaps[colormap]

    levels = list()
    for lvl in lvl_gm_line_data:
        levels.append(lvl)
    levels.sort()

    # ---- NOW WE DO THE PLOT BELOW ----------------------------------------------
    base_linewidth = linewidth

    for lvl in levels[::-1]:
        line_data = lvl_gm_line_data[lvl]
        for i in line_data:  # element # i

            lines = line_data[i]

            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                else:  # regular lines
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=base_linewidth, color=colors(lvl))

        base_linewidth *= 1.25

    # deal with title -----------------------------------------------
    if title is None:
        title = r"the multi-grid great mesh"
        plt.title(title)
    elif title is False:
        pass
    else:
        plt.title(title)

    if data_only:
        return fig
    else:
        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        else:
            from phyem.src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
        plt.close()
        return None
