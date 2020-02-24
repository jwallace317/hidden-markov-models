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
        self.t1 = np.zeros((len(state_space), length), dtype=np.float32)
        self.t2 = np.zeros((len(state_space), length), dtype=np.int8)

    # viterbi algorithm to compute most likely state path given observations
    def viterbi(self, observations):
        for i in range(len(self.state_space)):
            self.t1[i, 0] = self.transitions[i, len(
                self.state_space)] * self.emissions[observations[0] - 1, i]

        for j, observation in enumerate(observations[1:], start=1):
            max_values = np.zeros(
                (len(self.state_space), len(self.state_space)))
            for i in range(len(self.state_space)):
                for k in range(len(self.state_space)):
                    max_values[i, k] = self.emissions[observation - 1,
                                                      i] * self.transitions[i, k] * self.t1[k, j - 1]
                self.t1[i, j] = np.max(max_values[i, :])
                self.t2[i, j] = np.argmax(max_values[i, :])

        # print t1 and t2
        print('\nself.t1')
        print(self.t1)
        print('\nself.t2')
        print(self.t2)

        max_index = np.argmax(self.t1[:, self.length - 1])
        max_state = self.state_space[max_index]

        max_states_path = [max_state]
        for j, observation in zip(range(len(observations) - 1, 0, -1), observations[::-1]):
            prev_max_state_index = self.t2[max_index, j]
            prev_max_state = self.state_space[prev_max_state_index]
            max_states_path.append(prev_max_state)
            max_index = prev_max_state_index

        print('\nmax state path')
        print(max_states_path[::-1])

        return max_states_path[::-1]
