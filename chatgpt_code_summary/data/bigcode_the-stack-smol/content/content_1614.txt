from pathlib import Path


def dir_touch(path_file) -> None:
    Path(path_file).mkdir(parents=True, exist_ok=True)


def file_touch(path_file) -> None:
    p = Path(path_file)
    p.parents[0].mkdir(parents=True, exist_ok=True)
    p.touch()


def index_or_default(lst, val, default=-1):
    return lst.index(val) if val in lst else default


def print_info(logger, message):
    print(message)
    logger.info(message)
