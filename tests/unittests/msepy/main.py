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


stream = os.popen(rf'python {msepy_path}\m1n1.py')
output = stream.read()
print(output)

stream = os.popen(rf'python {msepy_path}\m2n2.py')
output = stream.read()
print(output)

stream = os.popen(rf'python {msepy_path}\m3n3.py')
output = stream.read()
print(output)


codifferential_tests = [
    rf'python {msepy_path}\codifferential_test.py 1 1 outer',
    rf'python {msepy_path}\codifferential_test.py 1 1 inner',
    rf'python {msepy_path}\codifferential_test.py 2 2 inner',
    rf'python {msepy_path}\codifferential_test.py 2 2 outer',
    rf'python {msepy_path}\codifferential_test.py 2 1 inner',
    rf'python {msepy_path}\codifferential_test.py 2 1 outer',
    rf'python {msepy_path}\codifferential_test.py 3 3 outer',
    rf'python {msepy_path}\codifferential_test.py 3 2 outer',
    rf'python {msepy_path}\codifferential_test.py 3 1 outer',
    rf'python {msepy_path}\codifferential_test.py 3 3 inner',
    rf'python {msepy_path}\codifferential_test.py 3 2 inner',
    rf'python {msepy_path}\codifferential_test.py 3 1 inner',
]

for _ in codifferential_tests:
    print(os.popen(_).read())


if __name__ == '__main__':
    # python tests/unittests/msepy/main.py
    pass
