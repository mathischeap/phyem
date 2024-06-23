# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM
from msehtt.static.mesh.great.visualize.matplot import MseHttGreatMeshVisualizeMatplot

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')


class MseHttElementsPartialMeshVisualizeMatplot(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def __call__(
            self,
            ddf=1,
            title=None,   # set title = False to turn off the title.
            data_only=False,
            **kwargs
    ):
        """"""
        outline_data = {}
        for i in self._elements:
            element = self._elements[i]
            outline_data[i] = element._generate_outline_data(ddf=ddf)
        mesh_data_Lines = COMM.gather(outline_data, root=MASTER_RANK)

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

        if title is None:
            title = f"${self._elements._tpm.abstract._sym_repr}$"
        else:
            pass

        if len(mn) == 1 and list(mn)[0] == (2, 2):  # all elements are 2d elements in 2d spaces.
            return MseHttGreatMeshVisualizeMatplot._plot_2d_great_mesh_in_2d_space(
                data, title=title, data_only=data_only, **kwargs
            )
        else:
            raise NotImplementedError()
