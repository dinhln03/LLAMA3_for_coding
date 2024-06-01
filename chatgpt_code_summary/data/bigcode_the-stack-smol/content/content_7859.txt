from setuptools import setup

# def readme():
#     with open('README.md') as f:
#         retun f.read()


setup(
    name = 'cypher',
    version = '0.2',
    author = 'shashi',
    author_email = 'skssunny30@gmail.com',
    description = 'Password Encryptor by suggesting wheather a password is strong or not',
    #long_description = readme(),
    long_description_content_type = 'text/markdown',
    url = "https://github.com/walkershashi/Cypher",
     classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], 
    lisence = 'MIT',
    packages = ['cypher'],
    )