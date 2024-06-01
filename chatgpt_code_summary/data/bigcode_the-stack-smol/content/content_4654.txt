from setuptools import find_packages, setup

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='ba_samples',
    package_dir={"": "src"},
    packages=find_packages('src'),
    version='0.1.0-dev0',
    description='Examples using ArcGIS Business Analyst with Python.',
    long_description=long_description,
    author='Joel McCune',
    license='Apache 2.0',
)
