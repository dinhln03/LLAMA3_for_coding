#!/usr/bin/env python2
from setuptools import setup
from setuptools import find_packages


setup(
    name="rover",
    version="0.1",
    description="Algorithm for risk and sensor quality aware sensor" +
    "coverage for quadrotors",
    author="Alex Wallar",
    author_email="wallarelvo@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    data_files=[
        (
            'config',
            ['configs/config.json'],
        )
    ]
)
