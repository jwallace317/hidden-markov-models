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
        self.t1 = np.zeros((len(state_space) - 1, length), dtype=np.float32)
        self.t2 = np.zeros((len(state_space) - 1, length), dtype=np.int8)

    # viterbi algorithm to compute most likely path
    def viterbi(self, sequence):
        states_dict = {}
        for i, state in enumerate(self.state_space[1:]):
            states_dict[state] = i
            self.t1[i, 0] = self.transitions[i, len(
                self.state_space) - 1] * self.emissions[sequence[0] - 1, states_dict[state]]

        for j, observation in enumerate(sequence[1:], start=1):
            max_values = np.zeros(
                (len(self.state_space) - 1, len(self.state_space) - 1))
            for i, src_state in enumerate(self.state_space[1:]):
                for k, dest_state in enumerate(self.state_space[1:]):
                    max_values[i, k] = self.emissions[observation - 1,
                                                      i] * self.transitions[i, k] * self.t1[k, j - 1]
                self.t1[i, j] = np.max(max_values[i, :])
                self.t2[i, j] = np.argmax(max_values[i, :])

        # print t1 and t2
        print('\nself.t1')
        print(self.t1)
        print('\nself.t2')
        print(self.t2)

        max_value = np.argmax(self.t1[:, self.length - 1])
        # print(f'\nmax value = { max_value }')
        max_state = self.state_space[max_value + 1]
        # print(f'\nmax state = { max_state }')

        max_values = [max_value]
        max_states = [max_state]
        for j, observation in zip(range(len(sequence) - 1, 0, -1), sequence[::-1]):
            prev_max_state_index = self.t2[max_value, j]
            # print(f'\nprev max state index = { prev_max_state_index }')
            prev_max_state = self.state_space[prev_max_state_index + 1]
            # print(f'\nprev max state = { prev_max_state }')
            max_states.append(prev_max_state)
            # print(f'\nmax prob state path = { max_states }')

        print('\nmax path')
        print(max_states[::-1])

        return max_states[::-1]
