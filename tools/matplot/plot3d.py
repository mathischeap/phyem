# -*- coding: utf-8 -*-
"""
"""
import matplotlib.pyplot as plt


def plot3d(x, y, v, title=None):
    """"""
    ax = plt.figure().add_subplot(projection='3d')
    for section_id in x:
        fx, fy, fv = x[section_id], y[section_id], v[section_id]
        ax.plot(fx, fy, fv, c='k')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    if title is None:
        pass
    elif title is False:
        pass
    else:
        plt.title(r'' + title)

    plt.show()
    return ax
