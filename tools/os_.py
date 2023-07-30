# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:17 PM on 7/30/2023
"""
from src.config import SIZE, RANK, MASTER_RANK, COMM
import os


def isfile(filename):
    """"""
    if SIZE == 1:

        return os.path.isfile(filename)

    else:
        if RANK == MASTER_RANK:
            ToF = os.path.isfile(filename)
        else:
            ToF = None

        return COMM.bcast(ToF, root=MASTER_RANK)


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


def rmdir(folder_name):
    if RANK == MASTER_RANK:
        os.rmdir(folder_name)
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
