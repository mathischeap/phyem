# -*- coding: utf-8 -*-
"""
Test the reduction and reconstruction for msehtt-adaptive-mesh.

mpiexec -n 4 python tests/msehtt/adaptive/trf_tests.py
"""

import numpy as np

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

N = 5
K = 5
c = 0.

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)

# ------------- implementation ---------------------------------------------------
msehtt, obj = ph.fem.apply('msehtt-a', locals())

tgm = msehtt.tgm()
msehtt.config(tgm)(
    # 'chaotic', element_layout=K, c=c
    'meshpy',
    points=(
        [0, 0],
        [0, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, 0],
    ),
    max_volume=0.2,
    ts=1,
    renumbering=True,
)

_mesh = obj['mesh']
msehtt.config(_mesh)(tgm, including='all')

msehtt.initialize()

assert _mesh.current.composition.num_global_elements == 276, _mesh.current.composition.num_global_elements


def refining_function(x, y):
    r""""""
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


msehtt.renew(trf={
    'rff': refining_function,
    'rft': [0.5,],
    'rcm': 'center',
})
assert _mesh.current.composition.num_global_elements == 404, _mesh.current.composition.num_global_elements


msehtt.renew(trf={
    'rff': refining_function,
    'rft': [0.5, 0.75],
    'rcm': 'center',
})
assert _mesh.current.composition.num_global_elements == 452, _mesh.current.composition.num_global_elements


msehtt.renew(trf={
    'rff': refining_function,
    'rft': [0.5,],
})
assert _mesh.current.composition.num_global_elements == 572, _mesh.current.composition.num_global_elements


msehtt.renew(trf={
    'rff': refining_function,
    'rft': [0.5, 0.75],
})
assert _mesh.current.composition.num_global_elements == 770, _mesh.current.composition.num_global_elements
