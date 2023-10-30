# -*- coding: utf-8 -*-
"""The .py file of Section Presetting.
"""

import sys
ph_path = ...  # customize the path to the dir that contains phyem.
sys.path.append(ph_path)

import phyem as ph

ph.config.set_embedding_space_dim(2)
ph.config.set_high_accuracy(True)
ph.config.set_pr_cache(False)
