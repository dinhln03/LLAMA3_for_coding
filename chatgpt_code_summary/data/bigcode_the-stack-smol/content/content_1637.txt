from setuptools import setup, find_packages

setup(
    name="squirrel-and-friends",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "emoji==0.5.4", "nltk==3.5", "pyspellchecker==0.5.4",
        "numerizer==0.1.5", "lightgbm==2.3.1",
        "albumentations==0.5.2", "opencv-python==4.5.1.48",
        "opencv-python-headless==4.5.1.48",
        "torch==1.7.1", "imgaug==0.4.0", 
        "numpy==1.19.5", "pandas==0.25.1",
        "tensorboard==2.4.1", "tensorboard-plugin-wit==1.8.0",
        "tensorflow-estimator==2.4.0", "tensorflow-gpu==2.4.1"
    ]
)
