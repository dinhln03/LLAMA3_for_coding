#!/usr/bin/env python
import os
from setuptools import setup
from setuptools import find_packages
import sys
from financialdatapy import __version__ as VERSION

# 'setup.py publish' shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    sys.exit()

description = 'Extract financial data of a company.'

with open('README.md', 'r') as f:
    long_description = f.read()

install_requires = [
    'pandas>=1.4.0',
    'requests>=2.27.1',
    'xmltodict>=0.12.0',
    'python-dotenv>=0.19.2',
    'beautifulsoup4>=4.10.0',
    'lxml>=4.7.1',
    'user_agent>=0.1.10',
]

project_urls = {
    'Source': 'https://github.com/choi-jiwoo/financialdatapy',
}

setup(
    name='financialdatapy',
    version=VERSION,
    author='Choi Jiwoo',
    author_email='cho2.jiwoo@gmail.com',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.10',
    keywords=['python', 'stock', 'finance'],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls=project_urls,
)
