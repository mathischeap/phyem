r"""The default configurations for the multi-grid version of the static msehtt implementation.
"""

# When we refine a mesh, each element is refined into uniform (on the reference coordinate level) elements
# on the next level.
default_refining_method = 'uniform'

# For 'uniform' refining: When we refine a mesh, an element in divided into alpha^d elements
# on the next level, i.e. the refined mesh. d is the dimensions of the mesh.
default_uniform_multigrid_refining_factor = 2  # alpha

# For 'uniform' refining: At most we can define how many levels of meshes.
# For example, when it is equal to 3, it means, besides the base mesh, we
# can at most have 2 levels of refined meshes.
default_uniform_max_levels = 3

assert isinstance(default_uniform_max_levels, int) and default_uniform_max_levels >= 1, \
    (f"at least, we have 1 level of grid, i.e., we only have a base mesh. "
     f"Then the multi-grid is equivalent to a non-multi-grid.")
