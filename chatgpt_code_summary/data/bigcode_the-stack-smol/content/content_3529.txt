import os


def iterate_files(directory):
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if _is_log_file(filename):
                yield os.path.join(root, filename)


def _is_log_file(filename):
    basename, extension = os.path.splitext(filename)
    return ((extension == '.xml') and (not basename.endswith('-assets')))
