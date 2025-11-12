# -*- coding: utf-8 -*-
r"""
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


class MseHttBoundarySectionPartialMeshVisualizeMatplot(Frozen):
    r""""""

    def __init__(self, boundary_section):
        r""""""
        self._boundary_section = boundary_section
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        mn = self._boundary_section.mn
        if mn == (2, 2):
            return self._plot_boundary_section_of_2d_mesh_in_2d_space(*args, **kwargs)

        elif mn == (3, 3):
            return self._plot_boundary_section_of_3d_mesh_in_3d_space(*args, **kwargs)

        elif mn == 'empty':  # this boundary has boundary sections at all.
            if RANK == MASTER_RANK:
                print(f"MseHtt boundary section: {self._boundary_section._tpm.abstract._sym_repr} is empty. "
                      f"Skip plotting.")
            else:
                pass

        else:
            raise NotImplementedError(mn)

    def _plot_boundary_section_of_2d_mesh_in_2d_space(
            self,
            ddf=1,
            linewidth=1.25,
            saveto=None,
            title=None,   # set title = False to turn off the title.
    ):
        r""""""
        fig = self._boundary_section._tgm.visualize.matplot(
            data_only=True, color='lightgray'
        )

        all_outline_data = {}

        for element_index___face_id in self._boundary_section:
            element_index, face_id = element_index___face_id
            element = self._boundary_section._tgm.elements[element_index]
            outline_data = element._generate_outline_data(ddf=ddf)[face_id]
            all_outline_data[element_index___face_id] = outline_data

        all_outline_data = COMM.gather(all_outline_data, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        for _ in all_outline_data:
            for element_index___face_id in _:
                x, y = _[element_index___face_id]
                plt.plot(x, y, linewidth=linewidth)

        # deal with title -----------------------------------------------
        if title is None:
            title = f"${self._boundary_section._tpm.abstract._sym_repr}$"
            plt.title(title)
        elif title is False:
            pass
        else:
            plt.title(title)

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_partial_mesh_boundary_section')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])

    def _plot_boundary_section_of_3d_mesh_in_3d_space(
            self,
            ddf=1,
            saveto=None,
            title=None,
    ):
        r""""""
        all_outline_data = {}

        for element_index___face_id in self._boundary_section:
            element_index, face_id = element_index___face_id
            element = self._boundary_section._tgm.elements[element_index]
            outline_data = element._generate_outline_data(ddf=ddf)[face_id]
            all_outline_data[element_index___face_id] = outline_data

        all_outline_data = COMM.gather(all_outline_data, root=MASTER_RANK)

        if RANK != MASTER_RANK:
            return
        else:
            pass

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        ax.tick_params(labelsize=12)
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$y$', fontsize=14)
        ax.set_zlabel(r'$z$', fontsize=14)

        for _ in all_outline_data:
            for element_index___face_id in _:
                x, y, z = _[element_index___face_id]
                ax.plot(x, y, z, linewidth=0.75, color='k')

        # deal with title -----------------------------------------------
        if title is None:
            title = f"${self._boundary_section._tpm.abstract._sym_repr}$"
            plt.title(title)
        elif title is False:
            pass
        else:
            plt.title(title)

        # save -----------------------------------------------------------
        if saveto is not None and saveto != '':
            plt.savefig(saveto, bbox_inches='tight')
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msehtt_partial_mesh_boundary_section')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
