# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.quadrature import quadrature
from phyem.tools.dds.region_wise_structured import DDSRegionWiseStructured
from phyem.tools._mpi import merge_dict
from phyem.src.config import RANK, MASTER_RANK, COMM

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.partial.elements.visualize.matplot import MseHttElementsPartialMeshVisualizeMatplot
from phyem.msehtt.static.mesh.partial.elements.visualize.vtk_ import ___vtk_m3n3_partial_mesh_elements___


class MseHttElementsPartialMeshVisualize(Frozen):
    r""""""

    def __init__(self, elements):
        r""""""
        self._elements = elements
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        mn = self._elements.mn
        if mn == (3, 3):
            return ___vtk_m3n3_partial_mesh_elements___(self._elements, *args, **kwargs)
        else:
            return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        r""""""
        if self._matplot is None:
            self._matplot = MseHttElementsPartialMeshVisualizeMatplot(self._elements)
        return self._matplot

    def _target(self, *args, **kwargs):
        r"""We plot the function on this mesh."""
        mn = self._elements.mn
        if mn == (2, 2):  # 2d mesh in 2d space.
            self._target_m2n2(*args, **kwargs)
        else:
            raise NotImplementedError()

    def _target_m2n2(self, function, sampling_factor=1, **kwargs):
        r"""We plot the function on this m2n2 mesh.

        Parameters
        ----------
        function
            Can be called like ``function(*coo)`` and return a list of components. If it returns a scalar, then
            it should be like `[s, ]`.

            ``coo`` are the coordinates in each mesh element.
        kwargs :
            kwargs sent to the visualizer (of dds-rws).

        Returns
        -------

        """
        density = int(13 * sampling_factor)
        if density < 7:
            density = 7
        elif density > 31:
            density = 31
        else:
            pass
        xi = quadrature(density, 'Gauss').quad_nodes
        xi, et = np.meshgrid(xi, xi, indexing='ij')

        x_dict = {}
        y_dict = {}
        comp_val_dict_list = None
        dtype = ''
        for i in self._elements:
            element = self._elements[i]
            x, y = element.ct.mapping(xi, et)
            x_dict[i] = x
            y_dict[i] = y
            val = function(x, y)
            if comp_val_dict_list is None:
                if len(val) == 1 and isinstance(val[0], np.ndarray):  # scalar
                    comp_val_dict_list = [dict(), ]
                    dtype = 'scalar'
                elif len(val) == 2 and isinstance(val[0], np.ndarray) and isinstance(val[1], np.ndarray):  # vector
                    comp_val_dict_list = [dict(), dict()]
                    dtype = 'vector'
                else:
                    # REMEMBER: for tensor, do not put val into 2d list.
                    raise NotImplementedError(f"NotImplemented for len({len(val)})-typed data.")
            else:
                pass

            if dtype == 'scalar':
                comp_val_dict_list[0][i] = val[0]
            elif dtype == 'vector':
                comp_val_dict_list[0][i] = val[0]
                comp_val_dict_list[1][i] = val[1]
            else:
                raise NotImplementedError()

        x_dict, y_dict = merge_dict(x_dict, y_dict)

        all_VAL_LIST = COMM.gather(comp_val_dict_list, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            total_val_dict_list = None
            for val_list in all_VAL_LIST:
                if val_list is None:
                    pass
                else:
                    if total_val_dict_list is None:
                        total_val_dict_list = [dict() for _ in range(len(val_list))]
                    else:
                        pass

                    for i in range(len(val_list)):
                        total_val_dict_list[i].update(val_list[i])

            dds = DDSRegionWiseStructured([x_dict, y_dict], total_val_dict_list)

            dds.visualize(**kwargs)
