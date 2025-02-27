# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt mesh built upon msepy 2d meshes.

mpiexec -n 4 python tests/msehtt/numbering.py
"""

import sys

ph_dir = './'  # customize it to your dir containing phyem
if ph_dir not in sys.path:
    sys.path.append(ph_dir)

import numpy as np

from msehtt.static.mesh.great.config.vtu import MseHttVtuInterface
from random import uniform, randint
from src.config import MASTER_RANK, RANK, COMM

import __init__ as ph


def fx(t, x, y):
    return np.sin(2.1268 * np.pi * x + 0.021) * np.cos(1.842 * np.pi * y) * np.exp(t)


def fy(t, x, y):
    return np.cos(1.16486 * np.pi * x) * np.sin(1.32654 * np.pi * y) * np.exp(t)


def fw(t, x, y):
    return np.sin(1.8421 * np.pi * x + 0.215462) * np.sin(2.232151 * np.pi * y + 0.0215) * np.exp(t)


vector = ph.vc.vector(fx, fy)
scalar = ph.vc.scalar(fw)


def ph_test(mesh_no):
    r""""""
    ph.config.set_embedding_space_dim(2)
    ph.config.set_high_accuracy(True)
    ph.config.set_pr_cache(False)

    manifold = ph.manifold(2)
    mesh = ph.mesh(manifold)

    Out0 = ph.space.new('Lambda', 0, orientation='outer')
    Out1 = ph.space.new('Lambda', 1, orientation='outer')
    Out2 = ph.space.new('Lambda', 2, orientation='outer')

    Inn0 = ph.space.new('Lambda', 0, orientation='inner')
    Inn1 = ph.space.new('Lambda', 1, orientation='inner')
    Inn2 = ph.space.new('Lambda', 2, orientation='inner')

    o0 = Out0.make_form(r'\tilde{\omega}^0', 'outer-form-0')
    o1 = Out1.make_form(r'\tilde{\omega}^1', 'outer-form-1')
    o2 = Out2.make_form(r'\tilde{\omega}^2', 'outer-form-2')

    i0 = Inn0.make_form(r'{\omega}^0', 'inner-form-0')
    i1 = Inn1.make_form(r'{\omega}^1', 'inner-form-1')
    i2 = Inn2.make_form(r'{\omega}^2', 'inner-form-2')

    ph.space.finite(3)

    # ------------- implementation ---------------------------------------------------
    msehtt, obj = ph.fem.apply('msehtt-s', locals())
    tgm = msehtt.tgm()
    if mesh_no == 0:
        msehtt.config(tgm)('crazy', element_layout=5, c=0, trf=1, ts=1)
    elif mesh_no == 1:
        msehtt.config(tgm)('chaotic', element_layout=20, c=0, ts=False)
    elif mesh_no == 2:
        points = [
                (-5, -1),
                (0, -1),
                (0, 0),
                (1, 0),
                (5, 0),
                (5, 1),
                (1, 1),
                (-5, 1),
        ]
        msehtt.config(tgm)('meshpy', ts=0, points=points, max_volume=0.1)
    elif mesh_no == 3:
        points = [
                (-5, -1),
                (-1, -1),
                (0, -1),
                (0, 0),
                (1, 0),
                (5, 0),
                (5, 1),
                (1, 1),
                (-1, 1),
                (-5, 1),
        ]
        msehtt.config(tgm)('meshpy', ts=1, points=points, max_volume=0.1)
    elif mesh_no in (4, 5):

        if RANK == MASTER_RANK:
            __ = [uniform(-0.1, 0.1) for _ in range(8)]

        else:
            __ = None

        a, b, c, d, e, f, g, h = COMM.bcast(__, root=MASTER_RANK)

        coo = {
            0: (0, 0),
            1: (0, 0.5),
            2: (0, 1),
            3: (0.5, 1),
            4: (1, 1),
            5: (1, 0.5),
            6: (1, 0),
            7: (0.5, 0),
            8: (0.5, 0.5),
            9: (0.25 + a, 0.25 + b),
            10: (0.25 + c, 0.75 + d),
            11: (0.75 + e, 0.75 + f),
            12: (0.75 + g, 0.25 + h),
            13: (1.5, 0),
            14: (1.5, 0.5),
            15: (1.5, 1),
            16: (2, 0),
            17: (2, 0.5),
            18: (2, 1),
        }

        connections = {
            '0t0': [0, 9, 1],
            '0t1': [1, 9, 10],
            (2, 'S'): [2, 1, 10],
            '3': [2, 10, 3],
            't4': [10, 8, 11],
            '5': [3, 10, 11],
            6: [11, 5, 4],
            7: [3, 11, 4],
            8: [11, 12, 5],
            '9': [8, 12, 11],
            10: [8, 9, 12],
            11: [10, 9, 8],
            12: [9, 7, 12],
            't13': [7, 6, 12],
            14: [12, 6, 5],
            15: [9, 0, 7],
            'q0': [5, 6, 13, 14],
            'q1': [5, 14, 15, 4],
            'q3': [13, 16, 17, 14],
            'q4': [15, 14, 17, 18],
        }

        cell_types = {}
        for e in connections:
            cell_types[e] = 5
        cell_types['q0'] = 9
        cell_types['q1'] = 9
        cell_types['q3'] = 9
        cell_types['q4'] = 9

        if RANK == MASTER_RANK:
            __ = [randint(0, 3) for _ in range(20)]
        else:
            __ = None
        __ = COMM.bcast(__, root=MASTER_RANK)

        CONNECTIONS = {}

        for i, e in enumerate(connections):
            nodes = connections[e]
            rolling = __[i]
            if e in ['q0', 'q1', 'q3', 'q4']:
                if rolling == 0:
                    pass
                elif rolling == 1:
                    nodes = [nodes[1], nodes[2], nodes[3], nodes[0]]
                elif rolling == 2:
                    nodes = [nodes[2], nodes[3], nodes[0], nodes[1]]
                elif rolling == 3:
                    nodes = [nodes[3], nodes[0], nodes[1], nodes[2]]
                else:
                    raise Exception()
            else:
                if rolling == 0:
                    pass
                elif rolling == 1:
                    nodes = [nodes[1], nodes[2], nodes[0]]
                elif rolling == 2:
                    nodes = [nodes[2], nodes[0], nodes[1]]
                else:
                    pass

            CONNECTIONS[e] = nodes

        vif = MseHttVtuInterface(coo, CONNECTIONS, cell_types)
        if mesh_no == 4:
            msehtt.config(tgm)(vif, ts=0)
        else:
            msehtt.config(tgm)(vif, ts=1)

    elif mesh_no == 6:

        type_dict = {
            0: 9,
            1: 9,
            2: 9,
            3: 9,
        }

        coo_dict = {
            'A': (-1, 0),
            'B': (0.11, -0.5),
            'C': (1, 0),
            'G': (-2, -2),
            'H': (-1.5, -2.5),
            'P': (1.3, -2.5),
            'Q': (2, -1.5),
            'M': (-1, 2),
            'N': (0.87, 2.107),
            'D': (0, 1.99),
        }

        map_dict = {
            0: ['M', 'A', 'B', "D"],
            3: ['D', 'B', 'C', 'N'],
            1: ['A', 'G', 'H', 'B'],
            2: ['B', 'P', 'Q', 'C'],
        }
        msehtt.config(tgm)({'indicator': 'tqr', 'args': (type_dict, coo_dict, map_dict)}, element_layout=4)

    else:
        raise NotImplementedError()

    msehtt_mesh = msehtt.base['meshes'][r'\mathfrak{M}']
    msehtt.config(msehtt_mesh)(tgm, including='all')

    # msehtt_mesh.visualize()

    fi1 = obj['i1']
    fo0 = obj['o0']
    fo1 = obj['o1']

    fo0.cf = scalar
    fo0[0].reduce()
    fo1.cf = vector
    fo1[0].reduce()
    fi1.cf = vector
    fi1[0].reduce()

    check_local_cochain(fo0.cochain.gathering_matrix, fo0[0].cochain)
    check_local_cochain(fo1.cochain.gathering_matrix, fo1[0].cochain)
    check_local_cochain(fi1.cochain.gathering_matrix, fi1[0].cochain)


def check_local_cochain(gm, cochain):
    r""""""
    cochain_dict = {}
    for e in gm:
        local_numbering = gm[e]
        local_cochain = cochain[e]

        for num, coc in zip(local_numbering, local_cochain):
            num = int(num)
            coc = float(coc)
            if num in cochain_dict:
                cochain_dict[num].append(coc)
            else:
                cochain_dict[num] = [coc]

    cochain_dict = COMM.gather(cochain_dict, root=MASTER_RANK)

    if RANK == MASTER_RANK:
        total_dict = {}
        for cd in cochain_dict:
            for num in cd:
                if num in total_dict:
                    total_dict[num].extend(cd[num])
                else:
                    total_dict[num] = cd[num]

        for num in total_dict:
            cochain = total_dict[num]
            if len(cochain) > 1:
                cochain = np.array(cochain)
                np.testing.assert_array_almost_equal(cochain, cochain[0])
            else:
                pass
    else:
        pass


if __name__ == '__main__':
    # mpiexec -n 4 python tests/msehtt/numbering.py
    ph_test(0)
    ph_test(1)
    ph_test(2)
    ph_test(3)
    ph_test(4)
    ph_test(5)
    ph_test(6)
