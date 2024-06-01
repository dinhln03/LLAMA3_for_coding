from setuptools import setup, find_packages

setup(
    name = "pierre",
    version = "1.0.0",
    py_modules = ["pierre"],
    install_requires = [
        "Click",
        "mistune",
        ],
    package_dir = {"": "pierre"},
    entry_points = {
        "console_scripts": [
            "pierre  =  pierre:main"
        ]
    },
)
        
