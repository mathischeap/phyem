# -*- coding: utf-8 -*-
r"""
"""
watcher = 'OSMANTHUS'

import time
import socket
import sys
import os
import pyautogui
import pyperclip
from random import random

if './' not in sys.path:
    sys.path.append('./')


from tools.miscellaneous.timer import MyTimer
from tools.miscellaneous.copy_a_image_to_clipboard import send_image_to_clipboard

this_file_dir = os.path.dirname(__file__)
watching_dir = this_file_dir + '/_watching'

quite_time = ['<080000', '>230000']

at_most_send_message_num = 5
at_most_send_img_num = 3

image_extensions = ['png', 'jpg']


emoji_setting = {

    'goodiebag':        [f"🛍️", f" 💨"],
    'DESKTOP-QL41DR8':  [f"🎛️", f" 💨"],    # STack: dgmp-cpc-laminar
    'DESKTOP-C6IR310':  [f"📟", f" 💨"],    # motor
    'starship':         [f"🚀", f" 💨"],
    'blackhole':        [f"🎆", f" 💨"],
    'durian':           [f"🥴", f" 💨"],
    'LAPTOP-Yi-MiBook': [f"🥾", f" 💨"],
    'Quasar':           [f"🌌", f" 💨"],

}


def watching():
    r""""""
    updating_times = 0
    print(f"Watching initializing! Leave Wechat active at the background and do not close this window...")
    time.sleep(1)
    print(f"Checking starts...")

    time.sleep(2)
    pyautogui.hotkey('ctrl', 'alt', 'w')
    time.sleep(2 + random())
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(2 + random())
    pyperclip.copy('Zanni')
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(2 + random())
    pyautogui.press('enter')
    time.sleep(2 + random())
    pyperclip.copy(f'{MyTimer.current_time()} PHYEM watching starts @ {watcher} 💦')
    time.sleep(0.1 + random())
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(2 + random())
    pyautogui.press('enter')
    time.sleep(2 + random())
    pyautogui.hotkey('ctrl', 'alt', 'w')
    print(f"Checking finished. Watching starts....")
    time.sleep(5)
    
    while 1:
        CTime = MyTimer.current_time_with_no_special_characters()
        c_time = CTime.split('_')[1]
        in_quite_time = False
        for qt in quite_time:
            if qt[0] == '>':
                if c_time > qt[1:]:
                    in_quite_time = True
                else:
                    pass
            elif qt[0] == '<':
                if c_time < qt[1:]:
                    in_quite_time = True
                else:
                    pass
            else:
                raise NotImplementedError(
                    f'quite_time={qt} format not understood.')

            if in_quite_time:
                break
            else:
                pass

        if in_quite_time:
            wait_time = 120
        else:
            wait_time = 60
            all_files = os.listdir(watching_dir)
            num_files = len(all_files)
            if num_files == 1:
                pass
            else:
                updating_times += 1
                sending_msg = 0
                sending_img = 0
                print(f"~~~ {updating_times}th updating @ {CTime}.")

                total_send_sequence = []

                message_2b_sent = []
                files_2b_delete = []

                all_images = list()

                all_files.sort()
                for filename in all_files:
                    if filename[:3] == 'WA_' and filename[-4:] == '.txt':
                        file_date = filename[3:18]
                        yyyymmdd, hhmmss = file_date.split('_')
                        dir_filename = watching_dir + f"/{filename}"
                        with open(dir_filename, 'r') as file:
                            message = file.read()
                        file.close()

                        message = message.split('-----TimeZone-----\n')
                        timezone = message[0]
                        info = message[1:]
                        info = ''.join(info)

                        message = f"{yyyymmdd}-{hhmmss}[{timezone}]: {info}"

                        message_2b_sent.append(message)
                        files_2b_delete.append(dir_filename)

                        total_send_sequence.append(message)

                    elif filename[:3] == 'WA_' and filename.split('.')[1] in image_extensions:

                        dir_filename = watching_dir + f"/{filename}"

                        total_send_sequence.append(
                            (
                                'image',
                                dir_filename
                            )
                        )

                        files_2b_delete.append(dir_filename)

                        all_images.append(dir_filename)

                    else:
                        pass

                if len(message_2b_sent) > at_most_send_message_num:
                    more_msg = True
                    R_message_2b_sent = message_2b_sent[-at_most_send_message_num:]
                else:
                    more_msg = False
                    R_message_2b_sent = message_2b_sent

                if len(all_images) > at_most_send_img_num:
                    more_img = True
                    R_images = all_images[-at_most_send_img_num:]
                else:
                    more_img = False
                    R_images = all_images

                if len(R_images) > 0 or len(R_message_2b_sent) > 0:
                    pyautogui.press('enter')
                    time.sleep(3 + random())
                    pyautogui.press('enter')
                    time.sleep(3 + random())

                    pyautogui.hotkey('ctrl', 'alt', 'w')
                    # time.sleep(2 + random())
                    # pyautogui.hotkey('ctrl', 'f')
                    # time.sleep(2 + random())
                    # pyperclip.copy('Zanni')
                    # pyautogui.hotkey('ctrl', 'v')
                    # time.sleep(2 + random())
                    # pyautogui.press('enter')

                    if more_msg and (not more_img):
                        pyperclip.copy(
                            f'[New] There are more than {at_most_send_message_num} '
                            f'MESSAGES left. Only send the last {at_most_send_message_num} ones.'
                        )
                        time.sleep(0.1 + random())
                        pyautogui.hotkey('ctrl', 'v')
                        time.sleep(0.5 + random())
                        pyautogui.press('enter')
                        time.sleep(1 + random())
                    elif (not more_msg) and more_img:
                        pyperclip.copy(
                            f'[New] There are more than {at_most_send_img_num} '
                            f'IMAGES left. Only send the last {at_most_send_img_num} ones.'
                        )
                        time.sleep(0.1 + random())
                        pyautogui.hotkey('ctrl', 'v')
                        time.sleep(0.5 + random())
                        pyautogui.press('enter')
                        time.sleep(1 + random())
                    elif more_msg and more_img:
                        pyperclip.copy(
                            f'[New] There are more MESSAGES and more IMAGES left. '
                            f'Only send the last ones.'
                        )
                        time.sleep(0.1 + random())
                        pyautogui.hotkey('ctrl', 'v')
                        time.sleep(0.5 + random())
                        pyautogui.press('enter')
                        time.sleep(1 + random())

                    else:
                        pass

                    for _2sent in total_send_sequence:
                        if isinstance(_2sent, str):
                            msg = _2sent
                            if msg in R_message_2b_sent:

                                emoji_s = [f"🤖", f" 💨"]
                                for host in emoji_setting:
                                    if host in msg:
                                        emoji_s = emoji_setting[host]
                                        break
                                    else:
                                        pass
                                pre_emoji = emoji_s[0]
                                end_emoji = emoji_s[1]

                                pyperclip.copy(pre_emoji + msg + end_emoji)
                                time.sleep(0.1 + random())
                                pyautogui.hotkey('ctrl', 'v')
                                time.sleep(0.5 + random())
                                pyautogui.press('enter')
                                time.sleep(1 + random())
                                sending_msg += 1
                            else:
                                pass

                            with open(watching_dir + f"/log", 'a') as file:
                                file.write(msg + '\n\n')
                            file.close()

                        if isinstance(_2sent, tuple) and isinstance(_2sent[0], str) and _2sent[0] == 'image':
                            image_dir = _2sent[1]
                            if image_dir in R_images:
                                send_image_to_clipboard(image_dir)
                                time.sleep(0.1 + random())
                                pyautogui.hotkey('ctrl', 'v')
                                time.sleep(0.5 + random())
                                pyautogui.press('enter')
                                time.sleep(1 + random())
                                sending_img += 1
                            else:
                                pass

                            with open(watching_dir + f"/log", 'a') as file:
                                file.write('IMG >>> ' + image_dir + '\n\n')
                            file.close()

                    for df in files_2b_delete:
                        os.remove(df)

                    time.sleep(2 + random())

                    pyautogui.hotkey('ctrl', 'alt', 'w')

                    time.sleep(1 + random())

                    # --- clean log if it is too long ----------
                    with open(watching_dir + f"/log", 'r') as file:
                        log_str = file.read()
                    file.close()
                    if len(log_str) > 5000:
                        left_str = log_str[-500:]
                        with open(watching_dir + f"/log", 'w') as file:
                            file.write(left_str)
                        file.close()
                    else:
                        pass
                    del log_str
                    print(f"\t --- sent {sending_msg} messages.")
                    print(f"\t --- sent {sending_img} images.")
                    print()

                else:  # there are files but not recognized, still pass.
                    pass

        time.sleep(wait_time)


if __name__ == '__main__':
    hostname = socket.gethostname()

    if hostname != watcher:
        pass
    else:
        watching()
