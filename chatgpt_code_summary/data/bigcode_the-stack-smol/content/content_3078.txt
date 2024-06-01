
from setuptools import setup

setup(name='mws',
      version='0.2',
      description='Multi window sender',
      url='https://github.com/TheWorldOfCode/MWS',
      author='TheWorldOfCode',
      author_email='dannj75@gmail.com',
      install_requires=[
          "python-daemon>=2.2.4",
          "python-xlib>=0.27"
                       ],
      license='BSD',
      packages=['mws'],
      zip_safe=False)
