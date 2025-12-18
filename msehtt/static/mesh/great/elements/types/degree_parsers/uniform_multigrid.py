# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from phyem.tools.quadrature import quadrature


def ___parse_uniform_multigrid_degree___(MG_degree, parameters):
    r"""
    Parameters
    ----------
    MG_degree
        For example,
            case 1: MG_degree = 'MG-N3'
                We use Lobatto nodes for N=3 on the most refined mesh. And we will make proper conforming
                nodes for the coarse meshes to keep the dofs conforming.

    parameters

    """
    assert parameters['method'] == 'uniform', f"We only accept uniform multigrid "

    rff = int(parameters['rff'])
    lvl = int(parameters['lvl'])
    max_levels = int(parameters['max_levels'])

    if rff == 2:
        return ___parse_uniform_multigrid_degree_rff2___(MG_degree, lvl, max_levels)
    else:
        raise NotImplementedError(rff)


def ___parse_uniform_multigrid_degree_rff2___(MG_degree, lvl, max_levels):
    r""""""

    if MG_degree[:4] == 'MG-N':
        N = int(MG_degree[4:])
        assert N >= 3, f"N={N} wrong. N must >= 3."

        if lvl == max_levels - 1:
            nodes = quadrature(N, 'Lobatto').quad_nodes
            return N, nodes
        else:
            _, base_nodes = ___parse_uniform_multigrid_degree_rff2___(MG_degree, lvl+1, max_levels)
            assert base_nodes[0] == -1 and base_nodes[-1] == 1, \
                f"the nodes on the upper level (the next more refined level): {base_nodes} is wrong!"

            if N % 2 == 0:
                nodes = split_into_parts_according_to_distribution(base_nodes, int(N / 2), False)
                nodes = np.concatenate([nodes, -nodes[:-1][::-1]])
            else:
                nodes = split_into_parts_according_to_distribution(base_nodes, N // 2 + 1, True)
                nodes_L = nodes[:-1]
                nodes_R = - nodes_L
                nodes_R = nodes_R[::-1]
                nodes = np.concatenate([nodes_L, nodes_R])
            BASE_NODES = (base_nodes + 1) / 2
            BASE_NODES = np.concatenate([
                BASE_NODES - 1, BASE_NODES[1:]
            ])
            for n in nodes:
                assert abs(min(np.abs(BASE_NODES - n)) - 0) < 1e-12, f"must be!"
            return N, nodes
    else:
        raise NotImplementedError()


def split_into_parts_according_to_distribution(base_nodes, num_parts, half_last):
    r""""""
    ratio = [i + 1 for i in range(num_parts)]
    if half_last:
        ratio[-1] /= 2
    else:
        pass
    ratio = np.array(ratio) / sum(ratio)
    check_nodes = [-1, ]
    for r in ratio:
        check_nodes.append(
            check_nodes[-1] + 2 * r
        )
    check_nodes = check_nodes[1:-1]
    nodes = [-1, ]
    current = 0
    for i, node in enumerate(base_nodes):
        if node > check_nodes[current]:
            nodes.append(base_nodes[i-1])
            current += 1
        else:
            pass

        if current == num_parts - 1:
            break
    nodes.append(1)
    return (np.array(nodes) + 1) / 2 - 1
