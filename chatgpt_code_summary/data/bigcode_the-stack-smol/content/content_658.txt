import logging

from croncierge import cmd_services

def log_cmd(cmd_response):
    logging.info(f"Command response:\n{cmd_response}")

def test_cmd_stdout():
    cmd = "python3 /home/maxim/projects/celecron/tests/test_croncierge/debug_cmd.py"
    log_cmd(cmd_services.run_cmd(cmd))


def test_cmd_stderr():
    cmd = "python3 tests/test_croncierge/debug_cmd_error.py"
    log_cmd(cmd_services.run_cmd(cmd))


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.DEBUG)
    ...