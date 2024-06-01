from setuptools import setup

setup(
    name='geompy',
    version='0.1.0',
    description='Tools for Euclidean Geometry.',
    url='https://github.com/qthequartermasterman/geometry',
    author='Andrew P. Sansom',
    author_email='AndrewSansom@my.unt.edu',
    license='MIT',
    packages=['geompy'],
    install_requires=[
        'sympy',
        'numpy',
        'networkx',
        'matplotlib',
        'scikit-image',
        'symengine',
        'methodtools'
    ],
    extras_require={
        'gym_environments':  ["gym"]
    },

    classifiers=[
        ''
    ],
)
