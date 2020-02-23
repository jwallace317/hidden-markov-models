# import necessary modules
import numpy as np
import pandas as pd


class HiddenMarkovModel():

    def __init__(self, emission_probability, transition_probability):
        self.emission_probability = emission_probability
        self.transition_probability = transition_probability

    def viterbi(self, sequence):
        previous = np.array([])
        for i in range(self.emission_probability.shape[1]):
            previous = np.append(
                previous, self.transition_probability[i, 2] * self.emission_probability[sequence[0] - 1, i])

        max_previous = np.max(previous)

        max_state = []
        max_probability = []
        for i in np.arange(2, len(sequence) + 1):

            current = np.array([])
            for j in range(self.emission_probability.shape[1]):
                current = np.append(current, self.transition_probability[])
