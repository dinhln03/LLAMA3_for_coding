#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Taerbit",
    version="0.0.1",
    author="Finn Torbet",
    author_email="finnt26@gmail.com",
    description="Package to process images through interpretability methods and then measure them against a binary mask segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Taerbit/EXP",
    packages=['src'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False, install_requires=['cv2', 'numpy', 'keras', 'pandas', 'matplotlib', 'seaborn', 'shap', 'pathlib'])