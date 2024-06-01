import os
import re
import subprocess
import tempfile


def backups(destination, prefix):
    name_re = re.compile(r'^{}(?:\.[0-9]+)?$'.format(prefix))

    def _key(name):
        return [int(char) if char.isdigit() else char
                for char in re.split(r'([0-9]+)', name)]

    paths = []

    for name in sorted(os.listdir(destination), key=_key):
        path = os.path.join(destination, name)

        if os.path.isdir(path) and name_re.match(name):
            paths.append(path)

    return paths


def rotate(paths):
    for _ in range(len(paths)):
        path = paths.pop()

        prefix, _, suffix = os.path.basename(path).partition('.')
        suffix = str(int(suffix or 0) + 1)
        rotated = os.path.join(os.path.dirname(path), prefix + '.' + suffix)

        os.rename(path, rotated)

        paths.insert(0, rotated)


def sync(rsync, source, destination, link_dest=None, exclude=None):
    command = [rsync, '-v', '-a', '--no-D', '--delete', '--ignore-existing']

    if link_dest:
        command.extend(['--link-dest', link_dest])

    exclude = exclude or []

    if exclude:
        command.extend([arg for path in exclude for arg in ['--exclude', path]])

    if os.path.isdir(source) and source[-1] != os.path.sep:
        source += os.path.sep

    command.extend([source, destination])

    subprocess.check_call(command)


def touch(path, times=None):
    with open(path, 'a'):
        os.utime(path, times)


def delete(rsync, directory):
    empty = tempfile.mkdtemp() + os.path.sep

    command = [rsync, '-v', '-r', '--delete', empty, directory]

    subprocess.check_call(command)
    os.rmdir(directory)
    os.rmdir(empty)
