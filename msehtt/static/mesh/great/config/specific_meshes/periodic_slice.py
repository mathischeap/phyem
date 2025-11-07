# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


def periodic_slice(element_layout, bounds, thickness, levels=2):
    r"""This is a 3d fully periodic domain. It is a slice whose `thickness` is along the third dimension.

    Returns
    -------
    element_type_dict :
    element_parameter_dict :
    element_map_dict :
    levels:
        How many levels of elements along the thickness-dimension?

    """
    x_bounds, y_bounds = bounds
    x0, x1 = x_bounds
    y0, y1 = y_bounds

    assert thickness > 0, f"thickness={thickness} wrong, it must be greater than 0."
    assert x0 < x1, f"x-low-bound must be lower than x-upper-bound."
    assert y0 < y1, f"y-low-bound must be lower than y-upper-bound."
    assert isinstance(levels, int) and levels >= 1, f'levels={levels} wrong. It should be a positive integer.'

    # -------- parse element_layout ------------------------------------------------------
    if isinstance(element_layout, (list, tuple)):
        pass
    else:
        element_layout = [element_layout, element_layout]

    assert len(element_layout) == 2, f"element_layout must be a list of two entries."

    K0, K1 = element_layout
    if isinstance(K0, int) and K0 > 0:
        num_elements_x = K0
        spacing_x = np.linspace(x0, x1, num_elements_x + 1)
    else:
        raise Exception()

    if isinstance(K1, int) and K1 > 0:
        num_elements_y = K1
        spacing_y = np.linspace(y0, y1, num_elements_y + 1)
    else:
        raise Exception()

    num_elements_z = levels
    spacing_z = np.linspace(- thickness / 2, thickness / 2, num_elements_z + 1)

    # -- make elements -----------------------------------------------------------------

    if num_elements_z == 1:
        nodes_numbering = - np.ones((num_elements_x + 1, num_elements_y + 1), dtype=int)
        part_numbering = np.arange(num_elements_x * num_elements_y).reshape((num_elements_x, num_elements_y), order='F')
        nodes_numbering[:num_elements_x, :num_elements_y] = part_numbering
        nodes_numbering[-1, :num_elements_y] = part_numbering[0, :num_elements_y]
        nodes_numbering[:, -1] = nodes_numbering[:, 0]

        assert -1 not in nodes_numbering, f"must be!"

        element_type_dict = {}
        element_parameter_dict = {}
        element_map_dict = {}

        for j in range(num_elements_y):
            for i in range(num_elements_x):
                parameters = {}
                element_index = i + j * num_elements_x
                parameters['origin'] = (spacing_x[i], spacing_y[j], - thickness / 2)
                parameters['delta'] = (spacing_x[i+1] - spacing_x[i], spacing_y[j+1] - spacing_y[j], thickness)

                element_map = [
                    int(nodes_numbering[i, j]),
                    int(nodes_numbering[i+1, j]),
                    int(nodes_numbering[i, j+1]),
                    int(nodes_numbering[i+1, j+1]),
                    int(nodes_numbering[i, j]),
                    int(nodes_numbering[i+1, j]),
                    int(nodes_numbering[i, j+1]),
                    int(nodes_numbering[i+1, j+1]),
                ]

                element_type_dict[element_index] = 'orthogonal hexahedron'
                element_parameter_dict[element_index] = parameters
                element_map_dict[element_index] = element_map

        return element_type_dict, element_parameter_dict, element_map_dict

    else:
        nodes_numbering = - np.ones((num_elements_x + 1, num_elements_y + 1, num_elements_z + 1), dtype=int)
        part_numbering = np.arange(
            num_elements_x * num_elements_y * num_elements_z
        ).reshape((num_elements_x, num_elements_y, num_elements_z), order='F')

        nodes_numbering[:num_elements_x, :num_elements_y, :num_elements_z] = part_numbering
        nodes_numbering[-1, :num_elements_y, :num_elements_z] = part_numbering[0, :, :]
        nodes_numbering[:, -1, :num_elements_z] = nodes_numbering[:, 0, :num_elements_z]
        nodes_numbering[:, :, -1] = nodes_numbering[:, :, 0]

        assert -1 not in nodes_numbering, f"must be!"

        element_type_dict = {}
        element_parameter_dict = {}
        element_map_dict = {}

        for k in range(num_elements_z):
            for j in range(num_elements_y):
                for i in range(num_elements_x):
                    parameters = {}
                    element_index = i + j * num_elements_x + k * num_elements_x * num_elements_y
                    parameters['origin'] = (spacing_x[i], spacing_y[j], spacing_z[k])
                    parameters['delta'] = (
                        spacing_x[i+1] - spacing_x[i],
                        spacing_y[j+1] - spacing_y[j],
                        spacing_z[k+1] - spacing_z[k]
                    )

                    element_map = [
                        int(nodes_numbering[i, j, k]),
                        int(nodes_numbering[i+1, j, k]),
                        int(nodes_numbering[i, j+1, k]),
                        int(nodes_numbering[i+1, j+1, k]),
                        int(nodes_numbering[i, j, k+1]),
                        int(nodes_numbering[i+1, j, k+1]),
                        int(nodes_numbering[i, j+1, k+1]),
                        int(nodes_numbering[i+1, j+1, k+1]),
                    ]

                    element_type_dict[element_index] = 'orthogonal hexahedron'
                    element_parameter_dict[element_index] = parameters
                    element_map_dict[element_index] = element_map

        return element_type_dict, element_parameter_dict, element_map_dict
