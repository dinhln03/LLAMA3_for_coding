from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'edu-lib'
LONG_DESCRIPTION = 'Libary zum erlernen der Grundstruktur.'

setup(
        name="mylibrary", 
        version=VERSION,
        author="Stephan Bökelmann",
        author_email="sb@gruppe.ai",
        scripts=[],
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 
        url="",
        
        keywords=['python', 'debugging'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: POSIX",
        ]
)
