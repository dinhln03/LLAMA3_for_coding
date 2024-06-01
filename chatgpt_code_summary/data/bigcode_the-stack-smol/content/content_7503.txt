import os
from typing import List

project_dir = os.path.abspath(os.path.dirname(__file__))
c_file_suffix = ('.c', '.cc', '.cpp')


def read_file(file_name: str) -> List[str]:
    _f = open(file_name, 'r')
    lines = _f.readlines()
    _f.close()
    return lines
