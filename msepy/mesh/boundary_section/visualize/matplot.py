# -*- coding: utf-8 -*-

from tools.frozen import Frozen

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')


class Matplot(Frozen):
    """"""

    def __init__(self, bs):
        """"""
        self._bs = bs
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
            title=True,  # True, False, None, or custom

    ):
        """"""
        mesh = self._bs._base
        mesh_data_Lines = mesh.visualize._generate_mesh_grid_data(sampling_factor=sampling_factor)
        bs_face_data = self._bs.visualize._generate_element_faces_of_this_boundary_section(sampling_factor)

        plt.rc('text', usetex=usetex)

        ndim = mesh.ndim  # n
        esd = mesh.esd    # m

        if esd in (1, 2):  # we use 2-d plot --------------------
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
                            plt.plot(segment, [0, 0], linewidth=linewidth, color='lightgray')
                            plt.scatter(segment, [0, 0], color='lightgray')

            elif ndim == esd == 2:

                # we first plot the base mesh -------------------------------
                for i in mesh_data_Lines:  # region # i
                    lines = mesh_data_Lines[i]
                    for j, axis_lines in enumerate(lines):
                        axis0, axis1 = axis_lines
                        if j == 0:
                            plt.plot(axis0, axis1, linewidth=linewidth, color='lightgray')
                        elif j == 1:
                            plt.plot(axis0.T, axis1.T, linewidth=linewidth, color='lightgray')
                        else:
                            raise Exception

                # we then can plot the elements of this boundary section -----

                for i in bs_face_data:
                    # the #`i` face locally numbered in this boundary section.
                    data_i = bs_face_data[i]
                    plt.plot(*data_i, linewidth=1.25 * linewidth)

            else:
                raise NotImplementedError(f"not implemented for {ndim}-d mesh in {esd}-d space.")

        else:  # 3d plot for 3d meshes  ----------------------------
            raise NotImplementedError()

        # deal with title -----------------------------------------------
        if title is True:  # use the default title
            sym_repr = self._bs.abstract._sym_repr
            plt.title(rf"${sym_repr}$")
        elif title is None or title is False:
            pass
        else:
            plt.title(title)

        # ------- config the output --------------------------------
        plt.tight_layout()
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _matplot_setting
            plt.show(block=_matplot_setting['block'])
        return fig
