# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from phyem.tools.frozen import Frozen
from phyem.tools.functions.space._2d.angle import angle
from phyem.tools.miscellaneous.geometries.m2n2 import StraightSegment2


class CoordinatedUndirectedGraph(Frozen):
    r""""""

    def __init__(self):
        r""""""
        self._coordinates_ = {}
        self._edges_ = {}
        self._freeze()

    @property
    def coordinates(self):
        r"""Return a dict whose key is a node name and its value is a tuple or list of floats that indicates the
        coordinates of this node.

        For example,

            self.coordinates = {
                0: [1.5, 2.1],
                1: [3.0, 5.1],
                'A': (0, -1),
                'o': (0, 0),
            }

            It means this graph has four nodes, and they are in 2d space. The coordinates of the first node, named
            0, are x = 1.5, y = 2.1

        """
        return self._coordinates_

    @property
    def edges(self):
        r"""Return a dict whose key is a node name and its value is a list of other nodes
        that connect to this node.

        For example:

            self.edge = {
                0: [1, 2, 3, 'A'],
                1: [0, 2, 3],
                2: [0, 1],
                3: [0, 1, 'A'],
                'A': [0, 3],
            }

            It means for example, the node 0 is connected to nodes 1, 2, 3, 'A'.

        """
        return self._edges_

    def add_node(self, node_name, node_coordinates):
        r""""""
        if node_name in self._coordinates_:
            raise Exception(f"Warning: node named {node_name} exists, please use another name.")
        else:
            assert all([isinstance(coo, (int, float))for coo in node_coordinates]), \
                f"coordinates={node_coordinates} illegal, entries must all be int or floats."

            self._coordinates_[node_name] = node_coordinates

    def add_edge(self, name_of_node0, name_of_node1):
        r""""""
        assert name_of_node0 in self._coordinates_, f"node named {name_of_node0} does not exist."
        assert name_of_node1 in self._coordinates_, f"node named {name_of_node1} does not exist."

        if name_of_node0 not in self._edges_:
            self._edges_[name_of_node0] = list()
        else:
            pass

        if name_of_node1 not in self._edges_:
            self._edges_[name_of_node1] = list()
        else:
            pass

        if name_of_node1 in self._edges_[name_of_node0]:
            pass
        else:
            self._edges_[name_of_node0].append(name_of_node1)

        if name_of_node0 in self._edges_[name_of_node1]:
            pass
        else:
            self._edges_[name_of_node1].append(name_of_node0)

    def _find_dimensions(self):
        r""""""
        dimensions_pool = []
        for node in self._coordinates_:
            coordinates = self._coordinates_[node]
            dimensions_pool.append(len(coordinates))

        if all([_ == dimensions_pool[0] for _ in dimensions_pool]):
            return dimensions_pool[0]
        else:
            return None  # unknown

    def visualize(self, **kwargs):
        r""""""
        ndim = self._find_dimensions()
        if ndim == 2:
            return self.___visualize_2d___(**kwargs)
        else:
            raise NotImplementedError()

    def ___visualize_2d___(
            self,

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

        # ------ plot nodes and edges ----------------------------------
        for node in self._coordinates_:
            x, y = self._coordinates_[node]
            plt.scatter(x, y)
            plt.text(x, y, node, color='b')

        plotted_edges = []
        skip_edges = 0
        plot_edges = 0

        for node in self._edges_:
            x0, y0 = self._coordinates_[node]
            connected_nodes = self._edges_[node]
            for c_node in connected_nodes:
                if (c_node, node) in plotted_edges:
                    skip_edges += 1
                else:
                    plot_edges += 1
                    x1, y1 = self._coordinates_[c_node]
                    plt.plot(
                        [x0, x1], [y0, y1], linewidth=linewidth, c=color,
                    )
                    plotted_edges.append((node, c_node))

        assert skip_edges == plot_edges, f"must be!"

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
            plt.close()
            return None

    # ------------ to pipe -------------------------------------------------------------------------------
    def to_pipe(self, diameter, inlets, outlets):
        r""""""
        if isinstance(diameter, (int, float)):  # all pipes have the same diameter.
            return self._to_homogeneous_pipe(diameter, inlets, outlets)
        else:
            raise NotImplementedError()

    def _to_homogeneous_pipe(self, diameter, inlets, outlets):
        r""""""
        ndim = self._find_dimensions()
        if ndim == 2:
            return self._to_2d_homogeneous_pipe(diameter, inlets, outlets)
        else:
            raise NotImplementedError()

    def _to_2d_homogeneous_pipe(self, diameter, inlets, outlets):
        r""""""
        assert self._find_dimensions() == 2, f"I works only for 2d space."
        # TODO: check if all angles are pi/2, if yes, we can call the following function.
        # TODO: Otherwise, call something else.
        return self._to_2d_perp_homogeneous_pipe(diameter, inlets, outlets)

    def _to_2d_perp_homogeneous_pipe(self, diameter, inlets, outlets):
        r""""""
        if not isinstance(inlets, (list, tuple)):
            inlets = [inlets, ]
        else:
            pass

        if not isinstance(outlets, (list, tuple)):
            outlets = [outlets, ]
        else:
            pass

        for node in inlets:
            assert node in self._edges_, f"node={node} of inlets is not a valid node."
            edges = self._edges_[node]
            assert len(edges) == 1, f"node={node} can only have 1 edge! Now it has {len(edges)} edges."

        for node in outlets:
            assert node in self._edges_, f"node={node} of outlets is not a valid node."
            edges = self._edges_[node]
            assert len(edges) == 1, f"node={node} can only have 1 edge! Now it has {len(edges)} edges."
            assert node not in inlets, f"a node cannot be inlet and outlet at the same time."

        POLYGONS = list()
        skip_edges = 0
        plot_edges = 0
        plotted_edges = list()

        INLET_EDGES = list()
        OUTLET_EDGES = list()

        node_in_out_lets_dict = {}

        for node in self._edges_:
            x0, y0 = self._coordinates_[node]
            connected_nodes = self._edges_[node]
            for c_node in connected_nodes:
                if (c_node, node) in plotted_edges:
                    skip_edges += 1
                else:
                    plot_edges += 1
                    x1, y1 = self._coordinates_[c_node]
                    rectangle, in_out_edge_dict = self._produce_rectangle_(
                        [x0, y0], [x1, y1], diameter, node, c_node,
                    )
                    plotted_edges.append((node, c_node))
                    POLYGONS.append(rectangle)

                    if node in inlets:
                        INLET_EDGES.append(in_out_edge_dict[node])
                        node_in_out_lets_dict[node] = in_out_edge_dict[node]
                    if c_node in inlets:
                        INLET_EDGES.append(in_out_edge_dict[c_node])
                        node_in_out_lets_dict[c_node] = in_out_edge_dict[c_node]

                    if node in outlets:
                        OUTLET_EDGES.append(in_out_edge_dict[node])
                        node_in_out_lets_dict[node] = in_out_edge_dict[node]
                    if c_node in outlets:
                        OUTLET_EDGES.append(in_out_edge_dict[c_node])
                        node_in_out_lets_dict[c_node] = in_out_edge_dict[c_node]

        assert skip_edges == plot_edges, f"must be!"
        gdf = gpd.GeoDataFrame({'geometry': POLYGONS})
        union_polygon = unary_union(gdf.geometry)
        # noinspection PyUnresolvedReferences
        points = np.array(union_polygon.exterior.coords)
        return points, INLET_EDGES, OUTLET_EDGES, node_in_out_lets_dict

    @staticmethod
    def _produce_rectangle_(x0y0, x1y1, diameter, node0, node1):
        r""""""
        angle01 = angle(x0y0, x1y1)
        angleA = angle01 + 0.75 * np.pi
        angleB = angle01 + 1.25 * np.pi
        angleC = angle01 - 0.25 * np.pi
        angleD = angle01 + 0.25 * np.pi

        x0, y0 = x0y0
        x1, y1 = x1y1

        xA = x0 + diameter * np.cos(angleA)
        yA = y0 + diameter * np.sin(angleA)

        xB = x0 + diameter * np.cos(angleB)
        yB = y0 + diameter * np.sin(angleB)

        xC = x1 + diameter * np.cos(angleC)
        yC = y1 + diameter * np.sin(angleC)

        xD = x1 + diameter * np.cos(angleD)
        yD = y1 + diameter * np.sin(angleD)

        poly = Polygon([(xA, yA), (xB, yB), (xC, yC), (xD, yD)])

        in_out_edge_dict = {
            node0: StraightSegment2((xA, yA), (xB, yB)),
            node1: StraightSegment2((xC, yC), (xD, yD)),
        }

        return poly, in_out_edge_dict
    # ===== TO PIPE ========================================================================================


if __name__ == '__main__':
    g = CoordinatedUndirectedGraph()
    g.add_node(0, [0, 0])
    g.add_node(1, [2, 0])
    g.add_node(2, [2, 1])
    g.add_node(3, [2, -3])
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(1, 3)

    # g.visualize()

    points, inlets, outlets, node_in_out_lets_dict = g.to_pipe(0.1, 0, 2)

    import __init__ as ph

    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(False)

    manifold = ph.manifold(2)
    mesh = ph.mesh(manifold)

    boundary = mesh.boundary()

    msehtt, obj = ph.fem.apply('msehtt-s', locals())
    tgm = msehtt.tgm()
    msehtt.config(tgm)('meshpy', ts=1, points=points, max_volume=0.01)
    # tgm.visualize()

    msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
    msehtt.config(msehtt_mesh)(tgm, including='all')

    msehtt_mesh.visualize()

    total_boundary = msehtt.base['meshes'][r'\eth\mathfrak{M}']
    msehtt.config(total_boundary)(tgm, including=msehtt_mesh)

    total_boundary.visualize()
