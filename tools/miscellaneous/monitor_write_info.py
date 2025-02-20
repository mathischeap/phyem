# -*- coding: utf-8 -*-
r"""
"""
from tools.miscellaneous.random_ import string_digits
from tools.miscellaneous.timer import MyTimer
import time

def ___write_info___(info_str):
    r""""""
    tz = time.strftime('%z')
    filename = MyTimer.current_time_with_no_special_characters() + f'_' + string_digits(8) + '.txt'
    filename = watching_dir + '/WA_' + filename

    assert isinstance(info_str, str), f"I can only write string!"

    with open(filename, 'w') as file:
        file.write(tz + '-----TimeZone-----\n')
        file.write(info_str)
    file.close()