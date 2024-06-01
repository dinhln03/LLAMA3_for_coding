import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faust_pydantic_validate",
    version="0.0.1",
    author="Alexey Kuzyashin",
    author_email="alex@rocketcompute.com",
    description="A small decorator for post data view validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kuzyashin/faust-pydantic-validate",
    packages=['faust_pydantic_validate'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
           "pydantic",
           "faust",
       ],
)
