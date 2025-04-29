# -*- coding: utf-8 -*-
r"""
Use outlook to email someone.
"""

from time import sleep
from random import random
import pyautogui
import pyperclip


def use_outlook_to_email_someone(to, title, message):
    r""""""
    pyautogui.hotkey('win')
    sleep(1 + random())
    for s in 'outlook':
        pyautogui.press(s)
        sleep(0.5 + random())
    pyautogui.press('enter')
    sleep(25 + random())
    pyautogui.hotkey('ctrl', 'n')
    sleep(5 + random())
    pyperclip.copy(to)
    pyautogui.hotkey('ctrl', 'v')
    sleep(1 + random())
    pyautogui.press('tab')
    sleep(1 + random())
    pyperclip.copy(title)
    pyautogui.hotkey('ctrl', 'v')
    sleep(1 + random())
    pyautogui.press('tab')
    sleep(1 + random())
    pyperclip.copy(message)
    pyautogui.hotkey('ctrl', 'v')
    sleep(1 + random())
    pyautogui.hotkey('ctrl', 'enter')
    sleep(10 + random())
    pyautogui.hotkey('alt', 'f4')


if __name__ == '__main__':
    use_outlook_to_email_someone('zhangyi_aero@hotmail.com', 'test2', 'message test 22')
