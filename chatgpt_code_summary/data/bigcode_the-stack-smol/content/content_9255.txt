import subprocess


def test_CLI_coffee_call():
    return_code = subprocess.call([
        # path is or should be in a setting somewhere
        '/home/pi/Programming/Automation/executables/rfoutlets_coffee.py',
        '1000',
        '-d',
        '0',
        '--test'
    ])
    assert return_code == 0
