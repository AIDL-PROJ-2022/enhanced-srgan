import os
import glob

from typing import List


def has_image_extension(file_path: str) -> bool:
    """Checks that file has image extension.
    Args:
        file_path: Path of the file to be checked.
    Returns:
        ``True`` if file has image extension, ``False`` otherwise.
    """
    return os.path.splitext(file_path)[1] in {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}


def images_in_dir(dir_path: str) -> List[str]:
    """Searches for all images in the directory.
    Args:
        dir_path: Path to the folder with images.
    Returns:
        List of images in the folder or its subdirectories.
    """
    # Retrieve all files in the folder or its subdirectories
    files = glob.iglob(f"{dir_path}/**/*", recursive=True)
    # Sort files and filter out non-image files
    return sorted(filter(has_image_extension, files))
