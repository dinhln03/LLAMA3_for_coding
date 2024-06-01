#!/usr/bin/env python
"""The setup script."""
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()



setup(
    author="Faris A Chugthai",
    author_email='farischugthai@gmail.com',
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'fatal_police_shootings=fatal_police_shootings.core:main',
        ],
    },

    license="MIT license",
    include_package_data=True,
    keywords='fatal_police_shootings',
    name='fatal_police_shootings',
    packages=find_packages(
        include=[
        'fatal_police_shootings', 'fatal_police_shootings.*'
    ]),
    test_suite='tests',

    url='https://github.com/farisachugthai/fatal_police_shootings',
    version='0.1.0',
    zip_safe=False,
)
