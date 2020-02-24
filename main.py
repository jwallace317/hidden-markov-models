# import necessary modules
import numpy as np
import pandas as pd

# import hidden markov model class
from hidden_markov_model import HiddenMarkovModel


# task main
def main():

    # read in provided csv data
    emissions_df = pd.read_csv('./lab3data/observationProbs.csv')
    transitions_df = pd.read_csv('./lab3data/transitionProbs.csv')
    sequences_df = pd.read_csv('./lab3data/testData.csv')

    # print the data frames
    print('\nemission probability data frame')
    print(emissions_df.head())
    print('\ntransition probability data frame')
    print(transitions_df.head())
    print('\ntest sequences data frame')
    print(sequences_df.head())

    # get observed space and state space
    observations = emissions_df['P(x|...)']
    states = list(emissions_df.columns)

    # print the state spcaes
    print('\nobservation space')
    print(observations)
    print('\nstate space')
    print(states)

    # convert the data frames to numpy arrays
    emissions = emissions_df.to_numpy()
    transitions = transitions_df.to_numpy()
    sequences = sequences_df.to_numpy()

    # trim the matrices of meta data
    emissions = np.delete(emissions, 0, axis=1)
    transitions = np.delete(transitions, 0, axis=1)
    sequences = np.delete(sequences, 0, axis=1)

    # print the trimmed matrices
    print('\ntrimmed emission matrix')
    print(emissions)
    print('\ntrimmed transition matrix')
    print(transitions)
    print('\ntrimmed sequence vector')
    print(sequences)

    max_prob_state_paths = []
    for sequence in sequences:
        print(f'\nsequence = { sequence }')

        # instantiate hidden markov model
        hmm = HiddenMarkovModel(observations,
                                states,
                                emissions,
                                transitions,
                                length=5)

        # viterbi algorithm
        max_prob_state_path = hmm.viterbi(sequence)

        max_prob_state_paths.append(max_prob_state_path)

    print('\nstate path of maximum probability given the sequence')
    print(max_prob_state_paths)


if __name__ == '__main__':
    main()
