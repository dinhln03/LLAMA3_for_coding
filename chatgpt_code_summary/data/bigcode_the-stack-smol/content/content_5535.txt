from setuptools import setup


setup(
    name="sphinx_rtd_theme_http",
    version="1.0.0",
    author="Ashley Whetter",
    url="https://github.com/AWhetter/sphinx_rtd_theme_http/browse",
    py_modules=["sphinx_rtd_theme_http"],
    install_requires=[
        "sphinx_rtd_theme",
    ],
    classifiers=[
        'Framework :: Sphinx',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: BSD License",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Topic :: Documentation',
        'Topic :: Software Development :: Documentation',
    ],
)
