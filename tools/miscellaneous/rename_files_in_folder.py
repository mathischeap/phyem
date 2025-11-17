r""""""
import os
import re


def contains_special_characters(input_string):
    pattern = re.compile(r"[@_!#$%^&*()<>?/|\\}{~:]")
    return True if pattern.search(input_string) else False


def rename_files_in_folder(folder_dir, prefix, extensions=None):
    r"""

    Parameters
    ----------
    folder_dir
    prefix :
        The files are renamed to be prefix + numbering.
    extensions :
        Only rename files whose extensions are in extensions

    Returns
    -------

    """
    assert os.path.exists(folder_dir), f"folder_dir={folder_dir} not exist."
    assert isinstance(prefix, str) and not contains_special_characters(prefix), \
        f"prefix={prefix} illegal, it must be a string without special characters."

    if extensions is None:
        extensions = 'all'
    else:
        if isinstance(extensions, str):
            extensions = [extensions, ]
        else:
            assert isinstance(extensions, (list, tuple)), f"pls put allowed extensions in a list or tuple."

    for count, filename in enumerate(os.listdir(folder_dir)):
        file_ext = os.path.splitext(filename)[1]
        if extensions == 'all' or file_ext in extensions:
            new_filename = f"{prefix}_{count+1}{file_ext}"
            os.rename(
                os.path.join(folder_dir, filename),
                os.path.join(folder_dir, new_filename)
            )


if __name__ == '__main__':
    rename_files_in_folder(r"C:\Users\zhang\Desktop\images", 'file')
