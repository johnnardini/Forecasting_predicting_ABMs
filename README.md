# Forecasting & Predicting ABM

Forecasting and predicting ABMs with BINNs README file

System Requirements:
====================
* Python 3
* High performance computing is recommended for ABM data generation and BINN training.
* ABM data generation was performed with 20 cores
* BINN training was performed using GPUs

Installation:
=============
Following [The Good Research Code Handbook](https://goodresearch.dev/setup), you can use pip to install the `src` package for this project. Once you have downloaded this code, you can install this package in the `src/`` directory by entering 
```
pip install -e .
```

Running the ABM and training BINN models:
=========================================
See the README.md files in `scripts/Data_generation` and `scripts/BINN_training/` to see how to run the ABMs and train BINN models to pre-computed ABM data, respectively. 

ABM forecasting and prediction:
===============================
ABM forecasting and prediction can be performed by running the jupyter notebooks located in `scripts/Forecasting/` and `scripts/predicting/`, respectively.


Contact:
========
Please contact John Nardini at nardinij@tcnj.edu if you have any questions.