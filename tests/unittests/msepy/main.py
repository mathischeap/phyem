# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:02 PM on 5/2/2023

# python tests/unittests/msepy/main.py
"""
import os
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.config import SIZE

assert SIZE == 1, f"msepy does not work with multiple ranks."

msepy_path = r'.\tests\unittests\msepy'

msepy_error_report = list()

stream = os.popen(rf'python {msepy_path}\m1n1.py')
output = stream.read()
print(output)
if output[-7:-1] == 'exit:0':
    msepy_error_report.append('msepy_m1n1')

stream = os.popen(rf'python {msepy_path}\m2n2.py')
output = stream.read()
print(output)
if output[-7:-1] == 'exit:0':
    msepy_error_report.append('msepy_m2n2')

stream = os.popen(rf'python {msepy_path}\m3n3.py')
output = stream.read()
print(output)
if output[-7:-1] == 'exit:0':
    msepy_error_report.append('msepy_m3n3')


if __name__ == '__main__':
    # python tests/unittests/msepy/main.py
    print(msepy_error_report)
