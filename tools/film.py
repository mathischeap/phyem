# -*- coding: utf-8 -*-
"""
making moves

pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:39 PM on 7/30/2023
"""
import cv2
import os
from src.config import RANK, MASTER_RANK
from tools.miscellaneous.timer import MyTimer
from tools.miscellaneous.random_ import string_digits


# noinspection PyUnresolvedReferences
def images(image_folder, video_name=None, duration=5, clean_images=False):
    """Make a move from images in the folder named `image_folder`.

    Each image will be a frame of the video. Images must be named in an increasing sequence
    start with 0 or any other positive integer. They will be played in an increasing sequence as
    well.

    :param image_folder:
    :param video_name:
    :param duration: The video will be of time `duration` seconds.
    :param clean_images: {bool,} Do we delete the used images when we have released the video?
    :return:
    """
    if RANK != MASTER_RANK:
        return
    all_files = os.listdir(image_folder)

    # ------- parse the video name --------------------------------
    if video_name is None:
        if 'video.avi' in all_files:
            video_name = image_folder + '/video_' + \
                         MyTimer.current_time_with_no_special_characters() + \
                         '_' + string_digits(5) + '.avi'
        else:
            video_name = image_folder + '/video.avi'
    else:
        pass

    # ------- select all legal images --------------------------------
    images = list()
    image_file_extensions = ('png', 'jpg', 'jpeg', 'df')
    for file in all_files:
        if '.' in file and file.split('.')[-1] in image_file_extensions:
            images.append(file)

    # ----- sort the images ------------------------------------------
    images.sort(key=lambda x: int(x.split('.')[0]))

    # -------- make the video ----------------------------------------
    total_frames = len(images)
    assert total_frames >= 1, f"There is no legal images in this folder."

    frame_per_second = int(total_frames / duration)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*'DIVX'),
        frame_per_second,
        (width, height),
    )

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    # --------- clear images -----------------------------------------
    if clean_images:  # do not easily clean images!
        for file in images:
            os.remove(image_folder + '/' + file)
    else:
        pass
