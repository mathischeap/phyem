# -*- coding: utf-8 -*-
r"""
$ mpiexec -n 4 python tests/msehtt/gathering_matrix_and_vectors.py
"""
import random
import numpy as np

from phyem.src.config import RANK, MASTER_RANK, SIZE, COMM
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from phyem.msehtt.tools.vector.static.global_gathered import MseHttGlobalVectorGathered

# ---- decide use how many gathering matrices ------------------------------------------------
if RANK == MASTER_RANK:
    num_gms = random.randint(2, 6)  # we use how many gathering matrices to do the test?
else:
    num_gms = 0

num_gms = COMM.bcast(num_gms, root=MASTER_RANK)

# ----------- make random element distribution -------------------------------------------
if RANK == MASTER_RANK:
    total_num_elements = random.randint(30, 85)
    random_distribution = random.sample(range(total_num_elements), total_num_elements)
    element_distribution = np.array_split(random_distribution, SIZE)
else:
    total_num_elements = None
    element_distribution = None

total_num_elements = COMM.bcast(total_num_elements, root=MASTER_RANK)
local_elements = COMM.scatter(element_distribution, root=MASTER_RANK)

# ------ make random gathering matrices and vectors --------------------------------------------------
GM = []
vectors = []
for _ in range(num_gms):  #
    if RANK == MASTER_RANK:
        total_num_dofs = random.randint(200, 400)
        vectors.append(np.random.random(total_num_dofs))
        random_numbering = random.sample(range(total_num_dofs), total_num_dofs)
        numbering = np.array_split(random_numbering, total_num_elements)
        numbering_dict = {}
        for e in range(total_num_elements):
            num_add_dofs = random.randint(0, 5)
            add_dofs = random.sample(random_numbering, num_add_dofs)
            numbering_dict[e] = list(numbering[e]) + add_dofs

    else:
        numbering_dict = None

    total_numbering_dict = COMM.bcast(numbering_dict, root=MASTER_RANK)

    local_numbering = {}
    for e in local_elements:
        local_numbering[e] = np.array(total_numbering_dict[e])

    gm = MseHttGatheringMatrix(local_numbering)
    GM.append(gm)

vectors = COMM.bcast(vectors, root=MASTER_RANK)

# ---- chain the gathering matrices to make a big one -----------------------------------------------
GM = MseHttGatheringMatrix(GM)

# --------- vector tests -----------------------------------------------------------------------------
VECTORS = []
for i, vec in enumerate(vectors):
    VECTORS.append(MseHttGlobalVectorGathered(vec, GM._gms[i]))

VECTOR = GM.merge(*VECTORS)
s_vectors = VECTOR.split()
M_VECTORS = GM.merge(*s_vectors)
S_vectors = M_VECTORS.split()

for vec, s_vec, S_vec in zip(vectors, s_vectors, S_vectors):
    np.testing.assert_array_almost_equal(vec, s_vec.V)
    np.testing.assert_array_almost_equal(vec, S_vec.V)
