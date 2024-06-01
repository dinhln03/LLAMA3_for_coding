try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Raster Vector Analysis',
    'author': 'Jan Kumor',
    'url': 'http://github.com/akumor/python-rastervectoranalysis',
    'download_url': 'http://github.com/akumor/python-rastervectoranalysis',
    'author_email': 'akumor@users.noreply.github.com',
    'version': '0.1',
    'install_requires': [''],
    'packages': ['rastervectoranalysis'],
    'scripts': [],
    'name': 'rastervectoranalysis'
}

setup(**config)
