# import necessary modules
import numpy as np
import pandas as pd

from hidden_markov_model import HiddenMarkovModel


# task main
def main():

    # read in provided data
    emission_probability = pd.read_csv('./lab3data/observationProbs.csv')
    transition_probability = pd.read_csv('./lab3data/transitionProbs.csv')
    data = pd.read_csv('./lab3data/testData.csv')

    print('emission probability')
    print(emission_probability.head())

    print('transition probability')
    print(transition_probability.head())

    print('test data')
    print(data.head())

    length = 5
    hmm = HiddenMarkovModel(length)
    hmm.viterbi()


if __name__ == '__main__':
    main()
