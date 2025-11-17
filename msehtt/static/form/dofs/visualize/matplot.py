# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.src.config import COMM, RANK, MASTER_RANK

if RANK == MASTER_RANK:
    import matplotlib.pyplot as plt
    import matplotlib

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
    })
    matplotlib.use('TkAgg')
else:
    pass


class MseHtt_StaticForm_Dofs_Visualize_Matplot(Frozen):
    r""""""
    def __init__(self, f):
        self._f = f
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        ind_str = self._f.space.str_indicator

        if ind_str == 'm2n2k0':
            return self.___m2n2k0___(*args, **kwargs)
        else:
            raise NotImplementedError(f"dof visualizer not implemented for {ind_str}.")

    def ___m2n2k0___(
        self,
        ddf=1,

        figsize=(8, 6),
        aspect='equal',
        usetex=True,

        labelsize=12,

        ticksize=12,
        xticks=None, yticks=None,
        minor_tick_length=4, major_tick_length=8,

        xlim=None, ylim=None,
        saveto=None,
        linewidth=0.5,
        color='lightgray',
        title=None,  # None or custom
        data_only=False,

        pad_inches=0
    ):
        r""""""
        elements = self._f.space.tpm.composition
        GDofs_coo_info = self._f.dofs.___global_dof_info___()

        outline_data = {}
        for i in elements:
            element = elements[i]
            outline_data[i] = element._generate_outline_data(ddf=ddf)
        mesh_data_Lines = COMM.gather(outline_data, root=MASTER_RANK)
        GDofs_coo_info = COMM.gather(GDofs_coo_info, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return None
        else:
            pass

        # from now on, we must be in the master rank.

        line_data = {}
        for _ in mesh_data_Lines:
            line_data.update(_)
        coo_info = {}
        for _ in GDofs_coo_info:
            coo_info.update(_)
        del mesh_data_Lines, GDofs_coo_info

        plt.rc('text', usetex=usetex)
        fig, ax = plt.subplots(figsize=figsize)
        # noinspection PyTypeChecker
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

        for i in line_data:  # element # i

            lines = line_data[i]

            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                elif isinstance(line_index, str) and line_index[:13] == 'internal_line':
                    # do not plot the internal lines
                    pass
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=linewidth, color=color)

        # ------------- plot dofs : scatter plot -------------------------------
        coo = []
        for dof in coo_info:
            ref_coo, phy_coo = coo_info[dof]
            coo.append(phy_coo)
        coo = np.array(coo).T
        plt.scatter(*coo, color='k', marker='o')

        # deal with title ------------------------------------------------------
        if title is None:
            title = rf"dofs of ${self._f.abstract._sym_repr}$"
            plt.title(title)
        elif title is False:
            pass
        else:
            plt.title(title)

        if data_only:
            return fig
        else:
            pass

        # save --------------------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])

        return None
