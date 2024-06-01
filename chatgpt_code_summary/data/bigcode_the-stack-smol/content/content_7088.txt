import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ChromedriverInstall",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['ChromedriverInstall = ChromedriverInstall.ChromedriverInstall:main']}
)




