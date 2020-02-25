# import necessary modules
import numpy as np


class LikelihoodSampler():

    def __init__(self, state_space, emissions, transitions):
        self.state_space = state_space
        self.emissions = emissions
        self.transitions = transitions

        self.index_dict = {}
        self.state_dict = {}
        for i, state in enumerate(self.state_space):
            self.index_dict[i] = state
            self.state_dict[state] = i

    # return the most likely state path given the observation
    def sample(self, observation, n):
        print(f'\nobservation = { observation }')

        weights = {}
        for i in range(n):
            states = []

            # calculate first state
            prior_prob_cold = self.transitions[self.state_dict['C'], 2]

            if prior_prob_cold > np.random.uniform():
                states.append(self.state_dict['C'])
            else:
                states.append(self.state_dict['H'])

            for j in range(1, len(observation)):
                conditional_prob = self.transitions[self.state_dict['C'],
                                                    states[j - 1]]

                if conditional_prob > np.random.uniform():
                    states.append(self.state_dict['C'])
                else:
                    states.append(self.state_dict['H'])

            # calculate weighted conditional probability given state path
            conditional_product = 1
            for i in range(len(observation)):
                conditional_product = conditional_product * \
                    self.emissions[observation[i] - 1, states[i]]

            if tuple(states) not in weights:
                weights[tuple(states)] = conditional_product
            else:
                weights[tuple(states)] = weights[tuple(states)] + \
                    conditional_product

            max_weight = 0
            max_state_path = []
            for state_path, weight in weights.items():
                if weight > max_weight:
                    max_weight = weight
                    max_state_path = state_path

            max_states = []
            for index in max_state_path:
                max_states.append(self.index_dict[index])

        return max_states
