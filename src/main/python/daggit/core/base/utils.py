import os
import errno
import contextlib
import sys
import mock
import yaml
import re


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


def parse_config(path=None, data=None, tag='!ENV'):
    pattern = re.compile('.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError('Either a path or data should be defined as input')