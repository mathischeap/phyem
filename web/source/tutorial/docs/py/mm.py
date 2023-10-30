# -*- coding: utf-8 -*-
r"""The .py file of Section Manifold & mesh.
"""

import sys
ph_path = ...  # customize the path to the dir that contains phyem.
sys.path.append(ph_path)

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)

manifold = ph.manifold(2)
mesh = ph.mesh(manifold)
ph.list_meshes()
