# setup.py file

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="my-lambdata-dspt5",  # the name that you will install via pip
    version="1.0",
    author="Devvin Kraatz",
    author_email="devvnet97@gmai.com   ",
    description="Made as an example while taking Lambda School's Data Science Course, come join it's highly recommended!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # required if using a md file for long desc
    # license="MIT",
    url="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME",
    # keywords="",
    packages=find_packages()  # ["my_lambdata"]
)
