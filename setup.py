from setuptools import setup, find_packages
#from distutils.core import Extension
import sys
import os

#sys.path.append("./joeynmt")

with open("requirements.txt", encoding="utf-8") as req_fp:
  install_requires = req_fp.readlines()

setup(
  name='speechjoey',
  version='0.0.1',
  description='JoeyNMT extended for speech processing',
  author='Lasse Becker-Czarnetzki',
  url='https://github.com/B-Czarnetzki/speechjoey',
  license='Apache License',
  install_requires=install_requires,
  packages=find_packages(exclude=[]),
  python_requires='>=3.5',
  project_urls={
    'Source': 'https://github.com/B-Czarnetzki/speechjoey',
    'Tracker': 'https://github.com/B-Czarnetzki/speechjoey/issues',
  },
  entry_points={
    'console_scripts': [
    ],
  }
)
