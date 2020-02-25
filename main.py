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
    observations_df = pd.read_csv('./lab3data/testData.csv')

    # print the data frames
    print('\nemission probability data frame')
    print(emissions_df.head())
    print('\ntransition probability data frame')
    print(transitions_df.head())
    print('\nobservations data frame')
    print(observations_df.head())

    # get observed space and state space
    observation_space = emissions_df['P(x|...)']
    state_space = list(emissions_df.columns)[1:]

    # print the state spcaes
    print('\nobservation space')
    print(observation_space)
    print('\nstate space')
    print(state_space)

    # convert the data frames to numpy arrays
    emissions = emissions_df.to_numpy()
    transitions = transitions_df.to_numpy()
    observations = observations_df.to_numpy()

    # trim the matrices of unnecessary meta data
    emissions = np.delete(emissions, 0, axis=1)
    transitions = np.delete(transitions, 0, axis=1)
    observations = np.delete(observations, 0, axis=1)

    # print the trimmed matrices
    print('\ntrimmed emission matrix')
    print(emissions)
    print('\ntrimmed transition matrix')
    print(transitions)
    print('\ntrimmed observations matrix')
    print(observations)

    max_state_paths = []
    for observation in observations:
        # trim observation
        observation = observation[observation != 0]

        # instantiate hidden markov model
        hmm = HiddenMarkovModel(observation_space,
                                state_space,
                                emissions,
                                transitions,
                                length=len(observation))

        # viterbi algorithm
        max_state_path = hmm.viterbi(observation)

        # append the max state path to list of max state paths
        max_state_paths.append(max_state_path)

    # print the max state paths given the observed sequences
    print('\n-------------------------MAX STATE PATHS GIVEN OBSERVED SEQUENCE-------------------------')
    print('{:^30s} {:^30s} {:^30s}'.format(
        'index', 'observed sequence', 'max state path'))
    for i, observation in enumerate(observations):
        observation = observation[observation != 0]
        print('{:^30d} {:^30s} {:^30s}'.format(
            i, str(observation), str(max_state_paths[i])))

    # likelihood sampling for approximate inference
    sample_size_convergence = []
    sample_state_paths = []
    sample_sizes = np.arange(1, 10)**2
    for i, observation in enumerate(observations):
        # trim observation
        observation = observation[observation != 0]

        # generate sample state paths for sample sizes
        for sample_size in sample_sizes:
            sample_state_path = hmm.sample(observation, sample_size)

            # if convergence is reached, break
            if max_state_paths[i] == sample_state_path:
                sample_size_convergence.append(sample_size)
                sample_state_paths.append(sample_state_path)
                break

    # print the likelihood sampling convergence results
    print('\n-----------------------------------------LIKELIHOOD SAMPLING CONVERGENCE RESULTS-----------------------------------------')
    print('{:^30s} {:^30s} {:^30s} {:^30s}'.format(
        'observation', 'sampled state path', 'max state path', 'sample size at convergence'))
    for observation, sample_path, max_path, sample_size in zip(observations, sample_state_paths, max_state_paths, sample_size_convergence):
        print('{:^30s} {:^30s} {:^30s} {:^30d}'.format(
            str(observation), str(sample_path), str(max_path), sample_size))


if __name__ == '__main__':
    main()
