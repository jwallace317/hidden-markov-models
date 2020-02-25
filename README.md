# CSE 5522: Lab 3 Hidden Markov Models and Viterbi's Algorithm

CSE 5522: Artificial Intelligence II
Lab 3: Hidden Markov Models and Viterbi's Algorithm
By: Jimmy Wallace

This repository contains the source code to complete CSE 5522: Artificial Intelligence II Lab 3 Hidden
Markov Models and Viterbi's Algorithm.

This lab aims to help students implement Viterbi's dynamic programming algorithm on a hidden markov
model. Given the provided data set, this repository implements Viterbi's algorithm to determine the
most likely sequence of discrete hidden states given an observed sequence of known random variables.

## Getting Started

Before getting started, ensure that you are using Python version 3.8.0

To get started with this lab, first fork this repository and install the required python dependencies
listed in the included requirements.txt file.

    pip install -r requirements.txt

Next, you can run the program with the following command

    python main.py

## Results

Running the main program will print the results of Viterbi's algorithm to the console. The first table
records the computed most likely state path given the observed sequence of known random variables. The
second table records the convergence results of computing the most likely state paths by sampling versus
the most likely state paths computed by Viterbi's Algorithm.
