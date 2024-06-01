from ..config import config
import logging
import os
import shutil
import subprocess
from subprocess import CalledProcessError, Popen
from time import sleep

logging.basicConfig(
    level=config.log_level, format='%(asctime)s | %(levelname)s | %(message)s')

PORT = config.port


def rm_gen_dir():
    try:
        shutil.rmtree('generated-tests')
    except(FileNotFoundError):
        pass


def deps_installed():
    return shutil.which('oatts') is not None and shutil.which('mocha') is not None


def run_oatts():
    logging.info('Running oatts tests...')
    if not deps_installed():
        logging.error('oatts is not installed! See the README.')
        exit(0)

    rm_gen_dir()

    try:
        subprocess.run(['oatts', 'generate',
                        '-w', 'generated-tests',
                        '-s', 'swagger/api.spec.yaml',
                        '--host', 'localhost:{}'.format(PORT),
                        '--customValuesFile', 'test/values.json'], check=True)
        subprocess.run(
            ['mocha', '--recursive', 'generated-tests'], check=True)
    except(CalledProcessError):
        logging.error('oatts tests failed!')
    finally:
        rm_gen_dir()


server_process = None


def start_server():
    global server_process
    env = os.environ.copy()
    env['INSIGHTS_CONNEXION_ENV'] = 'test'
    server_process = Popen(['pipenv', 'run', 'server'], env=env)
    sleep(5)


def test():
    try:

        logging.info('Testing...')
        start_server()
        run_oatts()
        logging.info('Testing is done')
    except() as err:
        logging.error(err)
    finally:
        server_process.terminate()
