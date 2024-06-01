#!/usr/bin/python3

from sys import version_info
from setuptools import setup

if version_info < (3, 5, 3):
    raise RuntimeError("aiopm requires Python 3.5.3+")

setup(
    name='aiopm',
    version='1.1',
    description='Async Postmark client (asyncio)',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX',
#        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Development Status :: 4 - Beta',
#        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
    ],
    author='Vitold Sedyshev',
    author_email='vit1251@gmail.com',
    maintainer=', '.join([
        'Vitold Sedyshev <vit1251@gmail.com>',
    ]),
    maintainer_email='aiopm@googlegroups.com',
    url='https://github.com/vit1251/aiopm',
    project_urls={
#        'CI: Travis': '...',
#        'Coverage: codecov': '...',
#        'GitHub: issues': '',
#        'GitHub: repo': '',
    },
    license='MIT',
    packages=['aiopm'],
    python_requires='>=3.5.3',
    install_requires=['aiohttp'],
    include_package_data=True,
)
