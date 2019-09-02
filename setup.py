# -*- coding: utf-8 -*-
# Part of [CCPCM Software](https://github.com/nghorbani/ccpcm)
# See LICENSE file for full copyright and licensing details.
# Implementation: Nima Ghorbani: nghorbani.github.io
#
# If you use this code please consider citing:
# Cross-Category Product Choice: A Scalable Deep-Learning Model (Sebastian Gabel and Artem Timoshenko)
#
#
# 2019.09.01
from setuptools import setup, find_packages

setup(name='ccpcm',
      version='0.2',
      packages = find_packages(),
      author=['Nima Ghorbani'],
      author_email=['nima.gbani@gmail.com'],
      maintainer='Nima Ghorbani',
      maintainer_email='nima.gbani@gmail.com',
      url='https://nghorbani.github.com/',
      description='Cross-Category Product Choice Model',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=['torch>=1.1.0', 'configer',  'tqdm', 'numpy'],
      dependency_links=[
          "https://github.com/nghorbani/configer/tarball/master#egg=configer"
      ],
      classifiers=[
          "Intended Audience :: Marketing",
          "Natural Language :: English",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Microsoft :: Windows",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",],
      )