from setuptools import setup

setup(
        name='yt-dl',
        version = "0.1.0",
        author = "Fernando Luiz Cola",
        author_email ="fernando.cola@emc-logic.com",
        license = "MIT",
        install_requires=[
            'Flask',
            'youtube-dl',
            ],
        )
