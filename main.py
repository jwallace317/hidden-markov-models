# import necessary modules
import numpy as np
import pandas as pd

from hidden_markov_model import HiddenMarkovModel


# task main
def main():

    # read in provided data
    emission_probability = pd.read_csv(
        './lab3data/observationProbs.csv').to_numpy()
    transition_probability = pd.read_csv(
        './lab3data/transitionProbs.csv').to_numpy()
    data = pd.read_csv('./lab3data/testData.csv').to_numpy()

    # remove the first column
    emission_probability = emission_probability[:, 1:]
    transition_probability = transition_probability[:, 1:]
    data = data[:, 1:]

    # print the head of the data frames
    print('emission probability')
    print(emission_probability)
    print('transition probability')
    print(transition_probability)
    print('test data')
    print(data)

    hmm = HiddenMarkovModel(emission_probability, transition_probability)

    for sequence in range(data.shape[0]):
        print(f'sequence { sequence } = { data[sequence, :] }')
        hmm.viterbi(data[sequence, :])


if __name__ == '__main__':
    main()
