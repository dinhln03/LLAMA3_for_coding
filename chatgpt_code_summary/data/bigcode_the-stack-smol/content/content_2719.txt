#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()

requirements = [
    'tweepy>=2.1',
    'pymongo>=2.8.0',
    'tendo>=0.0.18',
    'boto>=0.0.1',
    'nltk>=0.0.1',
    'zc.lockfile>=0.0.1',
    'flask>=0.0.1',
    'flask-bootstrap>=0.0.1'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='chattersum',
    version='0.1.0',
    description='test',
    author='Shane Eller',
    author_email='shane.eller@gmail.com',
    url='https://github.com/ellerrs/chattersum',
    packages=[
        'chattersum',
    ],
    package_dir={'chattersum':
                 'chattersum'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='chattersum',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests'
)
