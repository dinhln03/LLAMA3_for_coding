''' setup module
'''

from distutils.core import setup

# TEMPLATE
setup(
    name='mask-query-aide',
    version='0.0',
    description='python code to train ML for detecting people with masks',
    long_description=open('README.rst').read(),
    author='Christine Madden',
    license=open('LICENSE').read(),
    author_email='christine.m.madden19@gmail.com',
    packages=['mask_query_aide'],
    # python_requires="<3.8",
    install_requires=[
        "numpy==1.16.1",
        "pandas",
        "matplotlib",
        "opencv-python<=4.1.2.30",
        "keras==2.2.4",
        "tensorflow<2.0",
        "tensorflow-gpu<2.0",
        "imageai",
        "jupyterlab",
        "requests",
    ],
    entry_points={
        'console_scripts':
        [
            'mask_query_aide = mask_query_aide.__main__:main',
        ]
    }
)
