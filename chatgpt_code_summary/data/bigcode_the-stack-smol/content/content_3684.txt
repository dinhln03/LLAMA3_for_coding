import os

from setuptools import setup


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)

exec(compile(open('bevel/version.py').read(),
                  'bevel/version.py', 'exec'))

readme_md = filepath('README.md')

try:
    import pypandoc
    readme_rst = pypandoc.convert_file(readme_md, 'rst')
except(ImportError):
    readme_rst = open(readme_md).read()


setup(
    name="bevel",
    version="0.1.1",
    author="Ross Diener, Steven Wu, Cameron Davidson-Pilon",
    author_email="ross.diener@shopify.com",
    description="Ordinal regression in Python",
    license="MIT",
    keywords="oridinal regression statistics data analysis",
    url="https://github.com/ShopifyPeopleAnalytics/bevel",
    packages=[
        'bevel',
        ],
    long_description=readme_rst,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        ],
    install_requires=[
        "numpy>=1.13.3",
        "scipy>=1.0.0",
        "pandas>=0.21.0",
        "numdifftools>=0.9.20"
    ],
    package_data={
        "bevel": [
            "../README.md",
            "../LICENSE",
        ]
    },
)
