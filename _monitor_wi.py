# -*- coding: utf-8 -*-
r"""
"""
from tools.miscellaneous.random_ import string_digits
from tools.miscellaneous.timer import MyTimer
import time
import os
import socket

import shutil

this_file_dir = os.path.dirname(__file__)

watching_dir = this_file_dir + '/_watching'

image_extensions = ['png', 'jpg']


def ___write_info___(info_str):
    r""""""
    if os.path.isdir(watching_dir):
        tz = time.strftime('%z')
        filename = MyTimer.current_time_with_no_special_characters() + f'_' + string_digits(3) + '.txt'
        filename = watching_dir + '/WA_' + filename

        assert isinstance(info_str, str), f"I can only write string!"

        hostname = socket.gethostname()
        if hostname in info_str:
            pass
        else:
            info_str += f' < [{hostname}]'

        with open(filename, 'w') as file:
            file.write(tz + '-----TimeZone-----\n')
            file.write(info_str)
        file.close()
        time.sleep(1)  # make sure the filename are in a correct sequence.
    else:
        pass


def ___write_picture___(picture_path):
    r""""""
    if os.path.isdir(watching_dir):
        assert '.' in picture_path and picture_path.count('.') == 1, f"must have an extension."
        extension = picture_path.split('.')[1]
        assert extension in image_extensions, f"extension = {extension} illegal."
        filename = MyTimer.current_time_with_no_special_characters() + f'_' + string_digits(3) + '.' + extension
        filename = watching_dir + '/WA_' + filename
        shutil.copy(picture_path, filename)
        time.sleep(1)  # make sure the filename are in a correct sequence.
    else:
        pass


if __name__ == '__main__':
    # ___write_info___(f'This is a testing message made at {MyTimer.current_time()}')
    import matplotlib.pyplot as plt
    import numpy as np

    import random

    plot_no = random.randint(1, 5)
    if plot_no == 1:
        x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
        y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
        x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
        y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

        datasets = {
            'I': (x, y1),
            'II': (x, y2),
            'III': (x, y3),
            'IV': (x4, y4)
        }

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6),
                                gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
        axs[0, 0].set(xlim=(0, 20), ylim=(2, 14))
        axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12))

        for ax, (label, (x, y)) in zip(axs.flat, datasets.items()):
            ax.text(0.1, 0.9, label, fontsize=20, transform=ax.transAxes, va='top')
            ax.tick_params(direction='in', top=True, right=True)
            ax.plot(x, y, 'o')

            # linear regression
            p1, p0 = np.polyfit(x, y, deg=1)  # slope, intercept
            ax.axline(xy1=(0, p0), slope=p1, color='r', lw=2)

            # add text box for the statistics
            stats = (f'$\\mu$ = {np.mean(y):.2f}\n'
                     f'$\\sigma$ = {np.std(y):.2f}\n'
                     f'$r$ = {np.corrcoef(x, y)[0][1]:.2f}')
            bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
            ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,
                    transform=ax.transAxes, horizontalalignment='right')

        plt.suptitle(f"Test image made by {socket.gethostname()}\nat {MyTimer.current_time()}")

    elif plot_no == 2:
        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, r)
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

        ax.set_title(f"Test image made by {socket.gethostname()}\nat {MyTimer.current_time()}")

    elif plot_no == 3:
        np.random.seed(19680801)

        N = 100
        r0 = 0.6
        x = 0.9 * np.random.rand(N)
        y = 0.9 * np.random.rand(N)
        area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
        c = np.sqrt(area)
        r = np.sqrt(x ** 2 + y ** 2)
        area1 = np.ma.masked_where(r < r0, area)
        area2 = np.ma.masked_where(r >= r0, area)
        plt.scatter(x, y, s=area1, marker='^', c=c)
        plt.scatter(x, y, s=area2, marker='o', c=c)
        # Show the boundary between the regions:
        theta = np.arange(0, np.pi / 2, 0.01)
        plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
        plt.title(f"Test image made by {socket.gethostname()}\nat {MyTimer.current_time()}")

    elif plot_no == 4:
        theta = np.arange(0, 8 * np.pi, 0.1)
        a = 1
        b = .2

        for dt in np.arange(0, 2 * np.pi, np.pi / 2.0):
            x = a * np.cos(theta + dt) * np.exp(b * theta)
            y = a * np.sin(theta + dt) * np.exp(b * theta)

            Dt = dt + np.pi / 4.0

            x2 = a * np.cos(theta + Dt) * np.exp(b * theta)
            y2 = a * np.sin(theta + Dt) * np.exp(b * theta)

            xf = np.concatenate((x, x2[::-1]))
            yf = np.concatenate((y, y2[::-1]))

            p1 = plt.fill(xf, yf)

        plt.title(f"Test image made by {socket.gethostname()}\nat {MyTimer.current_time()}")

    elif plot_no == 5:
        np.random.seed(19680801)

        # Compute areas and colors
        N = 150
        r = 2 * np.random.rand(N)
        theta = 2 * np.pi * np.random.rand(N)
        area = 200 * r ** 2
        colors = theta

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

        plt.title(f"Test image made by {socket.gethostname()}\nat {MyTimer.current_time()}")

    else:
        raise NotImplementedError()

    test_pic_dir = 'test.png'
    plt.savefig(test_pic_dir)
    ___write_picture___(test_pic_dir)
    import os
    os.remove(test_pic_dir)
