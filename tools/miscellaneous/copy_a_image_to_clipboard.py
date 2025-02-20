# -*- coding: utf-8 -*-
r"""
"""
import win32clipboard
from PIL import Image
import io


def send_image_to_clipboard(image_path):
    r""""""
    image = Image.open(image_path)

    output = io.BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    finally:
        win32clipboard.CloseClipboard()
