# XORProblemPython

## Overview

This repository contains the Python code used for my IB Mathematics IA paper. 
The research aims to answer the question: **"How do different activation functions, specifically Sigmoid, Tanh and ReLU, affect the convergence rate and accuracy when solving the XOR problem in a neural network which is initialized using Xavier initialization?”**

## Features

- Solves the XOR problem with 1 hidden layer
- Switch between functions - tanh, sigmoid and ReLU
- Save training progress into a CSV file and plot it on graphs.
- CSV files with precomputed network training data for 10 seeds
- CSV files include loss, weight, bias and gradient values for 50k epochs
- Sequences prealigned for Homeobox genes coding sequences (Pseudogenes not included)

## Requirements

- Python 3.9
- Pytorch
- pandas
- NumPy
- matplotlib

## Usage

1. Using console clone the repository
```
git clone https://github.com/LauriSarap/XORProblemPython.git
```


2. Navigate to the repository folder
```
cd XORProblemPython
```
  

3. Install the required Python packages
```
pip install torch
pip install pandas
pip install matplotlib
pip install numpy
```
  

4. Run the main python script
```
python main.py
```


## Files

- `main.py`: The main Python script for running the training, data saving and plotting.
- `plotting_utilities.py`: Functions used for plotting the training progress
- `train_model.py`: Functions using Pytorch and training the network
