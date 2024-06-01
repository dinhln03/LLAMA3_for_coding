from setuptools import setup, find_packages

setup(
    name='wtouch',
    version='0.0.1',
    description='Create a file in current folder.',
    url='https://github.com/Frederick-S/wtouch',
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': [
            'wtouch = wtouch.main:main'
        ]
    },
    include_package_data=True,
    test_suite="tests"
)
