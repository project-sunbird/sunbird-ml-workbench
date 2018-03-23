import os, errno
import itertools

def create_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# def AbsPath(rel_path):
#     script_dir = os.path.dirname(os.path.dirname(__file__))  # <-- absolute dir of main
#     abs_file_path = os.path.join(script_dir, rel_path)
#     return abs_file_path


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def dicts_union(*dicts):
    return dict(itertools.chain.from_iterable(dct.items() for dct in dicts))


def dir_loc(dir_str):
    if dir_str[-1] != '/':
        dir_str = dir_str + '/'
    return dir_str


def get_parent_dir(location):
    return dir_loc(os.path.abspath(os.path.join(location, os.pardir)))

def normalize_path(cwd, path):
    if os.path.isabs(path):
        return path
    elif path.startswith('http'):
        return path
    elif path.startswith('s3://'):
        return path
    else:
        return os.path.normpath(os.path.join(cwd, path))
