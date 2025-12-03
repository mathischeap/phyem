# -*- coding: utf-8 -*-
r"""
"""

__version__ = '1.0.3'

__all__ = [
    'test',
    'config',

    'tools',

    'list_forms', 'list_spaces', 'list_meshes',
    'clear_forms',
    'samples',
    'manifold',
    'mesh',
    'space',
    'inner', 'dp',
    'wedge', 'Hodge',
    'd', 'exterior_derivative', 'trace',
    'codifferential',
    'time_derivative', 'ddt',

    'pde',
    'ode',
    'wf',

    'time_sequence',
    'constant_scalar',

    'fem',
    'vc',

    "iterator",

    "run",  # runner
    'rdr',  # runner data reader

    "save",
    "read",
    "read_tsf",

    "os",
    "film",

    "find_manifold",

    'vtk',  # save an object or some objects to a vtk file.
    'rws',  # save an object or some objects to a dds-rws-grouped file.

    'rref',

    'print_cache_log',   # to print a ph-cache file.
    'php',   # ph print
    'pk',    # a pickle wrapper for phyem
    'ws',    # write source code to

    'geometries',

]


def test():
    r"""Run the tests to valid your installation."""
    print(f"TEST TASK 1: tests for `msepy` implementation ...\n")
    # noinspection PyUnresolvedReferences
    import tests.msepy.main   # this automatically run all tests for the implementation `msepy`
    # so far, `msepy` is the only implementation released.


import phyem.src.config as config

from phyem.src.form.others import _list_forms as list_forms
from phyem.src.spaces.main import _list_spaces as list_spaces
from phyem.src.mesh import _list_meshes as list_meshes

from phyem.src.form.main import _clear_forms as clear_forms

import phyem.tests.samples.main as samples

from phyem.src.manifold import manifold
from phyem.src.manifold import find_manifold

from phyem.src.mesh import mesh

import phyem.src.spaces.main as space

from phyem.src.operators import inner, dp
from phyem.src.operators import wedge, Hodge, trace
from phyem.src.operators import d, exterior_derivative
from phyem.src.operators import codifferential
from phyem.src.operators import time_derivative
from phyem.src.operators import time_derivative as ddt

from phyem.src.pde import pde
from phyem.src.ode.main import ode
from phyem.src.wf.main import WeakFormulation as wf

from phyem.src.time_sequence import abstract_time_sequence as time_sequence
from phyem.src.form.parameters import constant_scalar

import phyem.src.fem as fem

import phyem.tools.vector_calculus as vc

from phyem.tools.runner import ParallelMatrix3dInputRunner as run  # runner date reader
from phyem.tools.runner import RunnerDataReader as rdr             # runner date reader

from phyem.tools.iterator.main import Iterator as iterator

from phyem.tools.save import save
from phyem.tools.read import read, read_tsf

import phyem.tools.os_ as os
import phyem.tools.film as film

# import phyem.tools

from phyem.tools.vtk_.main import vtk
from phyem.tools.dds.saving_api import _rws_grouped_saving as rws


def ws(file_dir, write_dir, source='source'):
    r""""""
    assert file_dir is not None and write_dir is not None, f"provide both file and and write dir"
    if config.RANK == config.MASTER_RANK:
        with open(file_dir, 'r') as file:
            source_code = file.read()
        file.close()
        with open(write_dir + rf"\{source}.py", 'w') as file:
            file.write(source_code)
        file.close()

    else:
        pass


___exist_signature___ = 'phyem exist 0'


def exist(file_dir=None, write_dir=None, source='source'):
    """"""
    if file_dir is None and write_dir is None:
        pass
    else:
        # when provided `file_dir` and `write_dir`,
        # we write the source code in file `file_dir` to `write_dir/{source}.py`
        assert file_dir is not None and write_dir is not None, f"provide both file and and write dir"
        ws(file_dir, write_dir, source=source)

    print(f"RANK#{config.RANK} ends smoothly.", flush=True)
    config.COMM.barrier()
    if config.RANK == config.MASTER_RANK:
        print(___exist_signature___)
    else:
        pass


from phyem.tools.miscellaneous.rref import rref

from phyem.tools.iterator.cache_reader import print_cache_log

from phyem.tools.miscellaneous.php import php

import phyem.tools.miscellaneous.pickle_ as pk

import phyem.tools.miscellaneous.geometries.main as geometries
