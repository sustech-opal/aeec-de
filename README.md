# Adaptive Differential Evolution based on Exploration and Exploitation Control (AEEC-DE)

[Click here to read the full paper online.](https://ieeexplore.ieee.org/abstract/document/9504876)

## Abstract

Search operator design and parameter tuning are essential parts of algorithm design. However, they often involve trial-and-error and are very time-consuming. A new differential evolution (DE) algorithm with adaptive exploration and exploitation control (AEEC-DE) is proposed in this work to tackle this challenge. The proposed method improves the performance of DE by automatically selecting trial vector generation strategies (both mutation and crossover operators) and dynamically generating the associated control parameter values. A probability-based exploration and exploitation measurement is introduced to estimate whether the state of each newly generated individual is in exploration or exploitation. The state of historical individuals is used to assess the exploration and exploitation capabilities of different generation strategies and parameter values. Then, the strategies and parameters of DE are adapted following the common belief that evolutionary algorithms (EAs) should start with exploration and then gradually change into exploitation. The performance of AEEC-DE is evaluated through experimental studies on a set of test problems and compared with several state-of-the-art adaptive DE variants.

## Keywords

Algorithm Configuration, Differential Evolution, Parameter Control, Exploration and Exploitation

## About this repository

### How to install

`pip install aeecde`

### How to use

`import aeecde`

### Tutorial

[Click here to read the full tutorial.](https://github.com/sustech-opal/aeec-de/blob/main/tutorial.ipynb)

## Citation

1. You may cite this work in a scientific context as:

    H. Bai, C. Huang and X. Yao, "Adaptive Differential Evolution based on Exploration and Exploitation Control", 2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. 41-48, doi: [10.1109/CEC45853.2021.9504876](https://doi.org/10.1109/CEC45853.2021.9504876)

2. Or copy the folloing BibTex file:

    ```latex
    @INPROCEEDINGS{AEECDE,
    author    = {Hao Bai and Changwu Huang and Xin Yao},
    title     = {Adaptive Differential Evolution based on Exploration and Exploitation Control},
    booktitle = {2021 IEEE Congress on Evolutionary Computation (CEC)},
    volume    = {},
    number    = {},
    pages     = {41-48},
    year      = {2021},
    url       = {https://ieeexplore.ieee.org/abstract/document/9504876},
    doi       = {10.1109/CEC45853.2021.9504876},
    }
    ```

3. Or [download the citation in RIS file (through IEEE Xplore).](https://ieeexplore.ieee.org/abstract/document/9504876)

## Related work

[Online algorithm configuration for differential evolution algorithm (OAC-DE)](https://pypi.org/project/oacde/)

## Contact

[Asst. Prof. Changwu HUANG](https://faculty.sustech.edu.cn/huangcw3/en/)
