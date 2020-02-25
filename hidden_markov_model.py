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

        self.index_dict = {}  # index to state dictionary
        self.state_dict = {}  # state to index dictionary
        for i, state in enumerate(self.state_space):
            self.index_dict[i] = state
            self.state_dict[state] = i

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

        max_index = np.argmax(self.t1[:, self.length - 1])
        max_state = self.state_space[max_index]

        max_states_path = [max_state]
        for j, observation in zip(range(len(observations) - 1, 0, -1), observations[::-1]):
            prev_max_state_index = self.t2[max_index, j]
            prev_max_state = self.state_space[prev_max_state_index]
            max_states_path.append(prev_max_state)
            max_index = prev_max_state_index

        return max_states_path[::-1]

    # return the most likely state path given the observation
    def sample(self, observation, n):
        sample_path_weights = {}
        for i in range(n):
            sample_path = []

            # calculate first state
            prior_prob = self.transitions[self.state_dict['C'], 2]
            if prior_prob > np.random.uniform():
                sample_path.append(self.state_dict['C'])
            else:
                sample_path.append(self.state_dict['H'])

            # calculate the remaining states
            for j in range(1, len(observation)):
                conditional_prob = self.transitions[self.state_dict['C'],
                                                    sample_path[j - 1]]
                if conditional_prob > np.random.uniform():
                    sample_path.append(self.state_dict['C'])
                else:
                    sample_path.append(self.state_dict['H'])

            # calculate weight of state path
            weight = 1
            for observed, state in zip(observation, sample_path):
                weight *= self.emissions[observed - 1, state]

            # add to weight dictionary
            if tuple(sample_path) not in sample_path_weights:
                sample_path_weights[tuple(sample_path)] = weight
            else:
                sample_path_weights[tuple(sample_path)] += weight

            # find max weight and max sample state path
            max_weight = 0
            max_sample_path = []
            for path, weight in sample_path_weights.items():
                if weight > max_weight:
                    max_weight = weight
                    max_state_path = path

            max_states = []
            for index in max_state_path:
                max_states.append(self.index_dict[index])

        return max_states
