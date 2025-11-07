# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

try:
    import meshpy.triangle as triangle
except ModuleNotFoundError:
    pass

from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface

from tools.functions.space._2d.distance import distance
import math


class MseHtt_API_2_MeshPy(Frozen):
    r""""""

    def __init__(self, **kwargs):
        r""""""

        if len(kwargs) == 1 and 'points' in kwargs:
            case = 0
            # case 0: only provide the outline points of the mesh.
            # it must be a 2d mesh, we can build the facets automatically.
            # All points are in a sequence around the domain.
            # the mesh will be triangular mesh.
            pass
        else:
            raise NotImplementedError()
        self._case = case
        self.___kwargs___ = kwargs
        self._freeze()

    def __call__(self, **kwargs):
        r"""build the mesh"""

        if self._case == 0:
            return self.___case_0_parser___(**kwargs)
        else:
            raise NotImplementedError

    def ___case_0_parser___(self, max_volume=None):
        r"""Only provided points. Then must be 2d mesh. All points are in a sequence around the domain.
        The mesh will be triangular mesh.

        The elements will be almost equally distributed. So no local refinement can be applied for this case.

        Parameters
        ----------
        max_volume :
            Determine the mesh density. The lower, the denser.

        """

        points = self.___kwargs___['points']
        seq = range(len(points))
        facets = ___round_trip_connect___(seq)

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)
        mesh = triangle.build(info, max_volume=max_volume)

        mesh_points = np.array(mesh.points)
        mesh_tris = np.array(mesh.elements)

        coo = {}

        renumbering = {}
        count = 0
        COO = {}
        for i, ___ in enumerate(mesh_points):
            coo[i] = ___
            coo_str = "%.9f" % ___[0] + ',' + "%.9f" % ___[1]
            if coo_str in renumbering:
                pass
            else:
                renumbering[coo_str] = count
                COO[count] = ___
                count += 1

        connections = {}
        cell_types = {}

        element_index = 0
        for connection in mesh_tris:
            assert len(connection) == 3, f"must be a triangle."
            element_connection = [int(_) for _ in connection]

            A = coo[element_connection[0]]
            B = coo[element_connection[1]]
            C = coo[element_connection[2]]

            # Square of lengths be a2, b2, c2
            a = distance(B, C)
            b = distance(A, C)
            c = distance(A, B)

            # length of sides be a, b, c
            a2 = a ** 2
            b2 = b ** 2
            c2 = c ** 2

            # From Cosine law
            alpha = (b2 + c2 - a2) / (2 * b * c)
            beta = (a2 + c2 - b2) / (2 * a * c)
            gamma = (a2 + b2 - c2) / (2 * a * b)

            if -1 < alpha < 1 and -1 < beta < 1 and -1 < gamma < 1:
                if alpha < -0.95 or alpha > 0.95 or beta < -0.95 or beta > 0.95 or gamma < -0.95 or gamma > 0.95:
                    alpha = math.acos(alpha)
                    beta = math.acos(beta)
                    gamma = math.acos(gamma)
                    # Converting to degree
                    alpha = alpha * 180 / math.pi
                    beta = beta * 180 / math.pi
                    gamma = gamma * 180 / math.pi
                    print(f"WARNING: very bad triangle found!, its inner angles are {alpha} {beta} {gamma}", flush=True)
                else:
                    pass

                coo_str_A = "%.9f" % A[0] + ',' + "%.9f" % A[1]
                coo_str_B = "%.9f" % B[0] + ',' + "%.9f" % B[1]
                coo_str_C = "%.9f" % C[0] + ',' + "%.9f" % C[1]
                TRUE_CONNECTION = [
                    renumbering[coo_str_A], renumbering[coo_str_B], renumbering[coo_str_C]
                ]
                cell_types[element_index] = 5  # all triangles
                connections[element_index] = TRUE_CONNECTION
                element_index += 1
            else:
                pass

        api2vtu = MseHttVtuInterface(
            COO, connections, cell_types,
            redistribute=True
        )
        return api2vtu


def ___round_trip_connect___(seq):
    r""""""
    result = []
    for i in range(len(seq)):
        result.append((seq[i], seq[(i + 1) % len(seq)]))
    return result
