# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, SIZE

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')


class MseHttGreatMeshVisualizeMatplot(Frozen):
    r""""""

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._freeze()

    def __call__(
        self,
        ddf=1,
        internal_grid=False,
        rank_wise_colored=False,
        quality=False,
        **kwargs
    ):
        r"""

        Parameters
        ----------
        ddf
        internal_grid
        rank_wise_colored
        quality

        """
        if quality:
            all_elements_quality_data = self._tgm.visualize._generate_element_quality_data()
            all_elements_quality_data = COMM.gather(all_elements_quality_data, root=MASTER_RANK)

            mesh_data_Lines = self._tgm.visualize._generate_element_outline_data(ddf=ddf)
            mesh_data_Lines = COMM.gather(mesh_data_Lines, root=MASTER_RANK)
        else:
            if internal_grid is False:
                mesh_data_Lines = self._tgm.visualize._generate_element_outline_data(ddf=ddf)
                mesh_data_Lines = COMM.gather(mesh_data_Lines, root=MASTER_RANK)
            else:
                if internal_grid is True:
                    internal_grid = 2
                else:
                    internal_grid = internal_grid
                assert isinstance(internal_grid, int) and internal_grid >= 0, \
                    f"internal_grid must be a integer indicating how many internal grid lines to be plotted."

                if rank_wise_colored:
                    internal_grid = 0
                else:
                    pass

                mesh_data_Lines = self._tgm.visualize._generate_element_outline_data(
                    ddf=ddf, internal_grid=internal_grid)
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

            if quality:
                qd = {}
                # noinspection PyUnboundLocalVariable
                for _ in all_elements_quality_data:
                    qd.update(_)

                self._plot_2d_great_mesh_in_2d_space_quality_(
                    data, qd, **kwargs
                )

            else:
                if rank_wise_colored:
                    return self._plot_2d_great_mesh_in_2d_space_rank_wise(
                        data,
                        element_distribution=self._tgm._element_distribution,
                        **kwargs
                    )
                else:

                    return self._plot_2d_great_mesh_in_2d_space(
                        data,
                        **kwargs
                    )
        else:
            raise NotImplementedError()

    @classmethod
    def _plot_2d_great_mesh_in_2d_space(
            cls,
            line_data,

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
            color='k',
            title=None,  # None or custom
            data_only=False,

            pad_inches=0
    ):
        r"""

        Parameters
        ----------
        line_data
        figsize
        aspect
        usetex
        labelsize
        ticksize
        xticks
        yticks
        minor_tick_length
        major_tick_length
        xlim
        ylim
        saveto
        linewidth
        color
        title : {str, bool, None}, optional
        data_only
        pad_inches

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
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    if line_index[14] == 'x':
                        _color = 'b'
                    elif line_index[14] == 'y':
                        _color = 'r'
                    else:
                        raise Exception()
                    plt.plot(x, y, linewidth=linewidth, color=_color)
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
            plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])

    @classmethod
    def _plot_2d_great_mesh_in_2d_space_quality_(
            cls, line_data, quality_data,

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
            color='k',
            title=None,  # None or custom
            data_only=False,

            pad_inches=0,
    ):
        r""""""
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

        colors = matplotlib.colormaps['OrRd']

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        for i in line_data:  # element # i

            lines = line_data[i]
            X_list = list()
            Y_list = list()

            x_sequence = []
            y_sequence = []
            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    plt.plot(x, y, linewidth=linewidth, color=color)
                    X_list.append(x)
                    Y_list.append(y)

                    x_sequence.extend(x)
                    y_sequence.extend(y)

            plt.fill(x_sequence, y_sequence, color=colors(1-quality_data[i]))

        # deal with title -----------------------------------------------
        if title is None:
            title = r"the great mesh: element quality"
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
            plt.savefig(saveto, bbox_inches='tight', pad_inches=pad_inches)
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_elements')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])

    @classmethod
    def _plot_2d_great_mesh_in_2d_space_rank_wise(
            cls,
            line_data,
            element_distribution=None,

            figsize=(12, 6),
            aspect='equal',
            usetex=True,

            labelsize=12,

            ticksize=12,
            xticks=None, yticks=None,
            minor_tick_length=4, major_tick_length=8,

            xlim=None, ylim=None,
            saveto=None,
            linewidth=0.75,
            color='k',
            title=None,  # None or custom
            data_only=False,

            pad_inches=0,
    ):
        r"""

        Parameters
        ----------
        line_data
        figsize
        aspect
        usetex
        labelsize
        ticksize
        xticks
        yticks
        minor_tick_length
        major_tick_length
        xlim
        ylim
        saveto
        linewidth
        color
        title : {str, bool, None}, optional
        data_only
        pad_inches

        Returns
        -------

        """
        plt.rc('text', usetex=usetex)
        # fig = plt.figure(figsize=figsize)
        # ax = plt.subplot2grid((2, 1), (0, 0))
        fig, AX = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [5, 1]})
        ax = AX[0]
        ax.set_aspect(aspect)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_xlabel(r"$x$", fontsize=labelsize)
        ax.set_ylabel(r"$y$", fontsize=labelsize)
        ax.tick_params(axis='both', which='both', labelsize=ticksize)
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        ax.tick_params(labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', direction='out', length=minor_tick_length)
        ax.tick_params(axis='both', which='major', direction='out', length=major_tick_length)

        colors = matplotlib.colormaps['tab20']

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        for i in line_data:  # element # i

            lines = line_data[i]

            x_sequence = []
            y_sequence = []
            for line_index in lines:
                if line_index == 'mn':
                    pass
                elif line_index == 'center':
                    pass
                elif isinstance(line_index, str) and line_index[:13] == 'internal_line':
                    raise Exception('no internal line allowed.')
                else:
                    line_xy_coo = lines[line_index]
                    x, y = line_xy_coo
                    ax.plot(x, y, linewidth=linewidth, color=color)
                    x_sequence.extend(x)
                    y_sequence.extend(y)

            in_rank = -1
            for rank in element_distribution:
                rank_elements = element_distribution[rank]
                if i in rank_elements:
                    in_rank = rank
                    break
                else:
                    pass
            fill_color = colors(in_rank)
            ax.fill(x_sequence, y_sequence, color=fill_color)

        # deal with title -----------------------------------------------
        if title is None:
            title = r"the great mesh"
            ax.set_title(title)
        elif title is False:
            pass
        else:
            ax.set_title(title)

        if data_only:
            return fig
        else:
            pass

        # -------- plot rank-legend --------------------------------------

        ax = AX[1]
        ax.set_aspect('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for rank in range(SIZE):
            color = colors(rank)
            ax.fill([0, 1, 1, 0, 0], [rank, rank, rank+0.5, rank+0.5, rank], color=color)
            ax.text(
                1.25, rank+0.25, f'{rank}',
                color='black', style='normal',
                ha='left', va='center', wrap=True
            )

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
