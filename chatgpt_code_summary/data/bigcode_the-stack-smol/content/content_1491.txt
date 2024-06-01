#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bcpy',
    version='0.1',
    author='Igor Neves Faustino',
    author_email='igornfaustino@gmail.com',
    url='https://github.com/igornfaustino/bcpy.git',
    description='library for BCI signal analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(),
    # entry_points={
    #     'console_scripts': ['forecastio = displayforecastio.app:run'],
    # }
)
