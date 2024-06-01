from setuptools import setup, find_packages

# declare these here since we use them in multiple places
_tests_require = [
    'pytest',
    'pytest-cov',
    'flake8',
]


setup(
    # package info
    name='cheapskate_bal',
    description='Cheapskate labs single/dual plane balancer',
    version='0.0.2',
    url='http://your/url/here',
    author='Kevin Powell',
    author_email='kevin@kevinpowell.guru',
    packages=find_packages(exclude=['tests', 'tests.*']),


    # scripts to install to usr/bin
    entry_points={
        'console_scripts': [
            'csbal=cheapskate_bal.cli:csbal_process',
            'csbal_s=cheapskate_bal.cli:csbal_single',
            'csbal_dinit=cheapskate_bal.cli:csbal_dual_init',
            'csbal_d=cheapskate_bal.cli:csbal_dual_iter'
        ]
    },


    # run time requirements
    # exact versions are in the requirements.txt file
    install_requires=[],

    # need this for setup.py test
    setup_requires=[
        'pytest-runner',
        
    ],

    # needs this if using setuptools_scm
    # use_scm_version=True,

    # test dependencies
    tests_require=_tests_require,
    extras_require={
        # this allows us to pip install .[test] for all test dependencies
        'test': _tests_require,
    }
)
