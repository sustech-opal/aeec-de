#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Adaptive Differential Evolution based on Exploration and Exploitation Control (AEEC-DE)

    Copyright 2022 Hao Bai, Changwu Huang and Xin Yao

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''
from setuptools import setup, find_packages

# read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# set PyPI information
setup(
    name='aeecde',
    version='0.0.1',
    url='https://github.com/sustech-opal/aeec-de',
    author='Hao Bai, Changwu Huang and Xin Yao',
    author_email='huangcw3@sustech.edu.cn',
    maintainer='Hao Bai',
    maintainer_email='hao.bai@insa-rouen.fr',
    license='Apache License Version 2.0, January 2004',
    platforms='py38-none-linux_x86_64',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'License :: Free For Educational Use',
    ],

    keywords='Algorithm Configuration, Differential Evolution, Parameter Control, Exploration and Exploitation',
    description='Adaptive Differential Evolution based on Exploration and Exploitation Control (AEEC-DE)',
    long_description=long_description,
    long_description_content_type='text/markdown',

    readme='README.md',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.17.3',
        'scipy>=1.2.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.1',
        'matplotlib>=2.2.3',
    ],
    zip_safe=False,
)