"""This script runs code quality checks on given Python files.

Note: This script assumes you use Poetry as your dependency manager.

Run the following in your terminal to get help on how to use this script:
```shell
poetry run python check_commit.py -h
```
"""


import argparse
import subprocess

from colorama import Fore, Style, deinit, init


def blue_bold(message: str) -> str:
    return f'{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}'


def light(message: str) -> str:
    return f'{Style.DIM}{message}{Style.RESET_ALL}'


def run_task(task_message: str, command: str) -> None:
    """Run a task in the shell, defined by a task message and its associated
    command."""
    print(blue_bold(task_message))
    print(light(f'$ {command}'))
    subprocess.call(command, shell=True)
    print()


if __name__ == '__main__':
    # initialise terminal colors
    init()

    # create parser
    parser = argparse.ArgumentParser(
        description=(
            f'Run code quality checks on the given Python files. By default '
            f'this script runs isort, Black and Flake8 successively but you '
            f'can use the parameters to selectively run some of these checks.'
        ),
        epilog=(
            'examples:\n'
            '\n'
            '  # run all checks on the my_package/ Python package\n'
            '  $ poetry run python check_commit.py my_package\n'
            '\n'
            '  # run Black and Flake8 on the la.py file and the foo/ folder\n'
            '  $ poetry run python check_commit.py -b -f8 la.py foo\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # add parser arguments
    parser.add_argument(
        '-i',
        '--isort',
        help='run isort on the given files',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--black',
        help='run Black on the given files',
        action='store_true',
    )
    parser.add_argument(
        '-f8',
        '--flake8',
        help='run Flake8 on the given files',
        action='store_true',
    )
    parser.add_argument(
        'files', type=str, nargs='+', help='list of files or directories',
    )

    # parse arguments
    args = parser.parse_args()

    # run checks
    run_all_checks = not args.isort and not args.black and not args.flake8
    files = ' '.join(args.files)

    if run_all_checks or args.isort:
        run_task(
            'Run import autosorting with isort...',
            f'poetry run isort -rc {files}',
        )

    if run_all_checks or args.black:
        run_task(
            'Run code formatting with Black...', f'poetry run black {files}',
        )

    if run_all_checks or args.flake8:
        run_task(
            'Run code linting with Flake8...', f'poetry run flake8 {files}',
        )

    # de-initialise terminal colors
    deinit()
