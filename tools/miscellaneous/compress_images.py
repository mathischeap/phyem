r""""""

from PIL import Image
import os

from phyem.tools.miscellaneous.timer import MyTimer


def compress_images_in_folder(input_folder_dir, output_folder_dir=None, quality=75):
    r"""

    Parameters
    ----------
    input_folder_dir
    output_folder_dir
    quality

    Returns
    -------

    """
    assert os.path.exists(input_folder_dir), f"input_folder_dir={input_folder_dir} not exist."
    if output_folder_dir is None:
        output_folder_dir = input_folder_dir + '_compressed_' + MyTimer.current_time_with_no_special_characters()
    else:
        pass

    if not os.path.exists(output_folder_dir):
        os.makedirs(output_folder_dir)
    else:
        pass

    for filename in os.listdir(input_folder_dir):
        img_path = os.path.join(input_folder_dir, filename)

        try:
            img = Image.open(img_path)
            output_path = os.path.join(output_folder_dir, filename)
            img.save(output_path, optimize=True, quality=quality)
        except:
            pass


if __name__ == '__main__':
    compress_images_in_folder(r"C:\Users\zhang\Desktop\images")
