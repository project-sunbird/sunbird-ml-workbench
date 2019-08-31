import os
import errno
import contextlib
import sys
import mock


def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class DummyFile(object):
    def write(self, x): pass

    # def flush(self): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    #sys.stdout = DummyFile()
    sys.stdout = mock.MagicMock()
    yield
    sys.stdout = save_stdout


def normalize_path(cwd, path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.normpath(os.path.join(cwd, path))


def get_as_list(string_or_list):
    if isinstance(string_or_list, list):
        return string_or_list
    else:
        return [string_or_list]
