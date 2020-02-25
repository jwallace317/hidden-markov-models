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

    # viterbi algorithm to compute the most likely state path given observations
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
        # create sample path weights dictionary
        sample_path_weights = {}
        for i in range(n):

            # create empty samplepath
            sample_path = []

            # calculate first state
            prior_prob = self.transitions[self.state_space.index('C'), 2]
            if prior_prob > np.random.uniform():
                sample_path.append(self.state_space.index('C'))
            else:
                sample_path.append(self.state_space.index('H'))

            # calculate the remaining states
            for j in range(1, len(observation)):
                transition_prob = self.transitions[self.state_space.index(
                    'C'), sample_path[j - 1]]
                if transition_prob > np.random.uniform():
                    sample_path.append(self.state_space.index('C'))
                else:
                    sample_path.append(self.state_space.index('H'))

            # calculate weight of state path
            weight = 1
            for observed, state in zip(observation, sample_path):
                weight *= self.emissions[observed - 1, state]

            # add to weight dictionary
            if tuple(sample_path) not in sample_path_weights:
                sample_path_weights[tuple(sample_path)] = weight
            else:
                sample_path_weights[tuple(sample_path)] += weight

            # compute max state path
            max_states = []
            for index in max(sample_path_weights, key=sample_path_weights.get):
                max_states.append(self.state_space[index])

        return max_states

    # compute the most likely state path given an observation using the forward backward algorithm
    def forward_backward(self, observation):
        # initial probability distribution
        initial_probs = self.transitions[0:len(
            self.state_space), len(self.state_space)]

        # transitions matrix
        transition_probs = self.transitions[0:2, 0:2]

        # forward pass
        forward = np.zeros((len(observation) + 1, len(self.state_space)))
        forward[0, :] = initial_probs
        for i in range(len(observation)):
            observation_probs = np.diag(self.emissions[observation[i] - 1, :])
            forward[i + 1, :] = np.linalg.multi_dot([forward[i, :], transition_probs, observation_probs]) / np.sum(
                np.linalg.multi_dot([forward[i, :], transition_probs, observation_probs]))

        # initial probability distribution
        initial_probs = np.ones((1, len(self.state_space)))

        # backward pass
        backward = np.zeros((len(observation) + 1, len(self.state_space)))
        backward[len(observation), :] = initial_probs
        for i in range(len(observation))[::-1]:
            observation_probs = np.diag(self.emissions[observation[i] - 1, :])
            backward[i, :] = np.linalg.multi_dot([backward[i + 1, :], transition_probs, observation_probs]) / np.sum(
                np.linalg.multi_dot([backward[i + 1, :], transition_probs, observation_probs]))

        # smoothing values
        s = np.zeros((len(observation) + 1, len(self.state_space)))
        for i in range(len(observation) + 1):
            s[i, :] = forward[i, :] * backward[i, :] / \
                np.sum(forward[i, :] * backward[i, :])

        # compute state path
        path = []
        for index in np.argmax(s, axis=1)[1:]:
            path.append(self.state_space[index])

        return path
