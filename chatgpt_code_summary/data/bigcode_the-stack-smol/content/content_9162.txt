import setuptools,subprocess,platform,os

with open("README.md","r",encoding="utf-8") as r:
  long_description=r.read()
URL="https://github.com/KoichiYasuoka/spaCy-ixaKat"

if platform.machine()=="x86_64" and os.name != "nt":
  subprocess.check_call(["spacy_ixakat/bin/download"])
else:
  raise OSError("spaCy-ixaKat only for 64-bit unix")

try:
  p=subprocess.check_output(["./packages.sh"])
except:
  p=subprocess.check_output(["spacy_ixakat/bin/packages"])
packages=p.decode("utf-8").rstrip().split("\n")

setuptools.setup(
  name="spacy_ixakat",
  version="0.6.4",
  description="ixaKat wrapper for spaCy",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url=URL,
  author="Koichi Yasuoka",
  author_email="yasuoka@kanji.zinbun.kyoto-u.ac.jp",
  license="MIT",
  keywords="ixaKat spaCy",
  packages=setuptools.find_packages(),
  package_data={"spacy_ixakat":packages},
  install_requires=["spacy>=2.2.2","deplacy>=1.9.9"],
  python_requires=">=3.6",
  classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
    "Topic :: Text Processing :: Linguistic",
    "Natural Language :: Basque",
  ],
  project_urls={
    "ixaKat":"http://ixa2.si.ehu.es/ixakat",
    "Source":URL,
    "Tracker":URL+"/issues",
  }
)
