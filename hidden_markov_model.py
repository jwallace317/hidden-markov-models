# import necessary modules
import numpy as np


# hidden markov model class
class HiddenMarkovModel():

    # initialize hidden markov model
    def __init__(self, observation_space, state_space, emissions, transitions, length):
        self.observation_space = observation_space  # observation space
        self.state_space = state_space  # state space
        self.emissions = emissions  # emission matrix
        self.transitions = transitions  # transmission matrix
        self.length = length  # length of hidden markov model
        self.t1 = np.zeros((len(state_space) - 1, length))
        self.t2 = np.zeros((len(state_space) - 1, length))

    # viterbi algorithm to compute most likely path
    def viterbi(self, sequence):
        states_dict = {}
        for i, state in enumerate(self.state_space[1:]):
            states_dict[state] = i
            self.t1[i, 0] = self.transitions[i, len(
                self.state_space) - 1] * self.emissions[sequence[0] - 1, states_dict[state]]

        # print t1 and t2
        print('\nself.t1')
        print(self.t1)
        print('\nself.t2')
        print(self.t2)

        for j, observation in enumerate(sequence[1:], start=1):
            max_observation = np.argmax(self.t1[:, j - 1])
            for i, state in enumerate(self.state_space[1:]):
                self.t1[states_dict[state], j] = self.emissions[observation - 1, states_dict[state]] * \
                    self.transitions[states_dict[state],
                                     max_observation] * self.t1[max_observation, j - 1]
                self.t2[states_dict[state], j] = np.argmax(
                    self.emissions[observation - 1, states_dict[state]] * self.transitions[states_dict[state], max_observation] * self.t1[max_observation, j - 1])

            # print t1 and t2
            print('\nself.t1')
            print(self.t1)
            print('\nself.t2')
            print(self.t2)
            input()

        return 1
