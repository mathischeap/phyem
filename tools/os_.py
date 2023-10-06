# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK
import os


def isfile(filename):
    """"""
    return os.path.isfile(filename)


def isdir(filename):
    """"""
    return os.path.isdir(filename)


def mkdir(folder_name):
    """"""
    if RANK == MASTER_RANK:
        if os.path.isdir(folder_name):
            pass
        else:
            os.mkdir(folder_name)
    else:
        pass


def remove(*file_names):
    """"""
    if RANK == MASTER_RANK:
        for file_name in file_names:
            os.remove(file_name)
    else:
        pass


def rmdir(*folder_names):
    if RANK == MASTER_RANK:
        for folder_name in folder_names:
            os.rmdir(folder_name)
    else:
        pass


def rename(old, new):
    if RANK == MASTER_RANK:
        os.rename(old, new)
    else:
        pass


def listdir(folder_name):
    """Return all filenames in the folder."""
    if RANK == MASTER_RANK:
        return os.listdir(folder_name)
    else:
        pass


def empty_dir(folder_name):
    """clean all files in a folder.

    Use this function very carefully. You do not want to delete all your important files accidentally.

    """
    if RANK == MASTER_RANK:
        files = listdir(folder_name)
        for file in files:
            os.remove(folder_name + '/' + file)
    else:
        pass
