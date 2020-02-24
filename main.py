# import necessary modules
import numpy as np
import pandas as pd

from hidden_markov_model import HiddenMarkovModel


# task main
def main():

    # read in provided csv data
    emissions_df = pd.read_csv('./lab3data/observationProbs.csv')
    transition_probability_df = pd.read_csv('./lab3data/transitionProbs.csv')
    sequences_df = pd.read_csv('./lab3data/testData.csv')

    # print the data frames
    print('emission probability')
    print(emissions_df.head())
    print('transition probability')
    print(transition_probability_df.head())
    print('test sequences')
    print(sequences_df.head())

    # get observed state space and the unobserved state space
    observed_states = emissions_df['P(x|...)']
    unobserved_states = list(emissions_df.columns)

    # print the state spcaes
    print(observed_states)
    print(unobserved_states)

    # convert the data frames to numpy arrays
    emissions = emissions_df.to_numpy()
    transition_probability = transition_probability_df.to_numpy()
    sequences = sequences_df.to_numpy()

    # print the trimmed emission and transition matrix
    transition_probability = np.delete(transition_probability, 0, axis=1)
    print(transition_probability)
    emissions = np.delete(emissions, 0, axis=1)
    print(emissions)
    sequences = np.delete(sequences, 0, axis=1)
    print(sequences)

    # instantiate hidden markov model
    hmm = HiddenMarkovModel(observed_states,
                            unobserved_states,
                            emissions,
                            transition_probability,
                            length=5)

    hmm.viterbi(sequences[0, :])


if __name__ == '__main__':
    main()
