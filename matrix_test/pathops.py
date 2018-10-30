import os
import shutil
from six import string_types

def get_filename(path):
    """
    Returns the filename without extension from path provided
    """
    return os.path.splitext(os.path.basename(path))[0]

def file_must_exist(path):
    """
    Checks that the path specified exists
    If not or the path is a directory then raises and IOError
    """
    # Check that path is a string value
    if not isinstance(path, string_types):
        raise TypeError("File path argument is not a string")
    if not os.path.isfile(path):
        raise IOError("File doesn't exist: ", path)
    else:
        return True


def dir_must_exist(path):
    """
    Checks that the directory exists.
    If not then tries to create the directory.
    If this fails then an IOError is raised.
    """
    if not isinstance(path, string_types):
        raise TypeError("Directory path argument is not a string")
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise IOError("Directory can't be created: ", path)

    return path


def listdir_nohidden(path):
    """
    List all files and subdirectories of a directory that aren't hidden.
    """
    dir_must_exist(path)
    return [os.path.join(path, i) for i in os.listdir(path) if not i.startswith('.')]


def delete_if_exists(path):
    """
    Trys to delete the file at the path specified if it exists.
    Returns True on succesful deletion else returns False.
    Either way there won't be a file at that location after running.
    """
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError:
        return False
    return True
