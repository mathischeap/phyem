# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, SIZE
from tools.frozen import Frozen
from msehy.py._2d.mesh.elements.main import MseHyPy2MeshElements
from _MPI.generic.py._2d_unstruct.mesh.elements.main import MPI_Py_2D_Unstructured_MeshElements

_global_element_distribution_cache = {
    'number base elements': -1,
    'master loading factor': -1.,
    'distribution': []
}


class Generic_Elements_Maker(Frozen):
    """"""

    def __init__(self, serial_generic_mesh):
        """

        Parameters
        ----------
        serial_generic_mesh
        """
        if RANK == MASTER_RANK:
            assert serial_generic_mesh.__class__ is MseHyPy2MeshElements, \
                f"Must have a serial msehy-py2 mesh elements in the master"
        else:
            assert serial_generic_mesh is None, f"we must receive None in non-master cores."
        sgm = serial_generic_mesh
        if RANK == MASTER_RANK:
            type_dict, vertex_dict, vertex_coordinates, same_vertices_dict \
                = sgm._make_generic_element_input_dict(sgm._indices)

            # -- distribute elements into ranks below: ---------------------------
            # scheme 1) A naive scheme
            # The elements are distributed according to the base elements.
            # This of course is not good because if the local refinement is happening in a single core,
            # the loading the that single core will be huge.
            indices_in_base_element = sgm.indices_in_base_element
            num_base_elements = len(indices_in_base_element)
            base_elements_distribution = self._make_base_element_distribution(num_base_elements)
            current = 0
            b_e_d = list()
            for dis in base_elements_distribution:
                b_e_d.append(
                    range(current, current+dis)
                )
                current += dis

            element_distribution = list()
            for s in range(SIZE):
                element_distribution.append(
                    list()
                )
            for index in type_dict:

                if isinstance(index, str):
                    element = int(index.split('=')[0])
                else:
                    element = index

                for rank, RANGE in enumerate(b_e_d):
                    if element in RANGE:
                        element_distribution[rank].append(index)
                    else:
                        pass
            # ====================================================================================
        else:
            type_dict, vertex_dict, vertex_coordinates, same_vertices_dict = {}, {}, {}, {}
            element_distribution = None
        self._inputs = type_dict, vertex_dict, vertex_coordinates, same_vertices_dict
        self._element_distribution = element_distribution
        self._freeze()

    def __call__(self):
        """"""
        return MPI_Py_2D_Unstructured_MeshElements(
            *self._inputs,
            element_distribution=self._element_distribution,
        )

    @staticmethod
    def _make_base_element_distribution(num_base_elements, master_loading_factor=0.5):
        """

        Parameters
        ----------
        num_base_elements :
        master_loading_factor :
            It must be in [0, 1], when it is lower, the master core loading is lower.

        Returns
        -------

        """
        if (_global_element_distribution_cache['number base elements'] == num_base_elements and
            _global_element_distribution_cache['master loading factor'] == master_loading_factor):
            return _global_element_distribution_cache['distribution']
        else:
            if SIZE == 1:
                base_elements_distribution = [num_base_elements, ]
            else:
                assert 0 <= master_loading_factor <= 1, \
                    f"master_loading_factor={master_loading_factor} wrong, it must be in (0, 1]."
                master_num = int((num_base_elements / SIZE) * master_loading_factor)
                base_elements_distribution = [master_num, ]
                r_num_elements = num_base_elements - master_num
                size = SIZE - 1
                remaining_numbers = [
                    r_num_elements // size + (1 if x < r_num_elements % size else 0) for x in range(size)
                ][::-1]
                base_elements_distribution += remaining_numbers

            _global_element_distribution_cache['number base elements'] = num_base_elements
            _global_element_distribution_cache['master loading factor'] = master_loading_factor
            _global_element_distribution_cache['distribution'] = base_elements_distribution

            return base_elements_distribution
