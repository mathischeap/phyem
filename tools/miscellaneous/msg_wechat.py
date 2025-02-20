# -*- coding: utf-8 -*-
r"""
"""
try:
    import pyautogui
    import pyperclip
except ModuleNotFoundError:
    pass

import time


def ___send___(msg):
    r""""""
    pyperclip.copy(msg)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.25)
    pyautogui.press('enter')
    time.sleep(0.25)


def send_msg(*messages, friend='Zanni'):
    r""""""
    # noinspection PyBroadException
    try:
        time.sleep(0.5)
        pyautogui.hotkey('ctrl', 'alt', 'w')
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'f')
        pyperclip.copy(friend)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(1)
        pyautogui.press('enter')
        for msg in messages:
            ___send___(msg)
        time.sleep(0.5)
        pyautogui.hotkey('ctrl', 'alt', 'w')
    except:
        pass


if __name__ == '__main__':
    # send a message (or messages) to someone.
    send_msg('Hello 555')
