# import necessary modules
import numpy as np


# hidden markov model class
class HiddenMarkovModel():

    # initialize hidden markov model
    def __init__(self, observed_states, unobserved_states, emissions, transitions, length):
        self.observed_states = observed_states
        self.unobserved_states = unobserved_states
        self.emissions = emissions
        self.transitions = transitions
        self.length = length
        self.t1 = np.zeros((len(unobserved_states) - 1, length))
        self.t2 = np.zeros((len(unobserved_states) - 1, length))

    def viterbi(self, sequence):
        unobserved_dict = {}
        for i, unobserved in enumerate(self.unobserved_states[1:]):
            unobserved_dict[unobserved] = i
            self.t1[i, 0] = self.transitions[i, len(
                self.unobserved_states) - 1] * self.emissions[sequence[0] - 1, unobserved_dict[unobserved]]
        print(unobserved_dict)
        print(self.t1)
        input()

        for i, observed in enumerate(sequence[1:], start=1):
            print(f'i = { i }')
            for j, unobserved in enumerate(self.unobserved_states[1:]):
                print(f'observed = { observed } unobserved = { unobserved }')
                print(self.t1[j, i - 1])
                print(self.emissions[observed, unobserved_dict[unobserved]])
                # self.t1[j, i] = self.t1[j, i - 1] *
                # self.emissions[observed, unobserved_dict[unobserved]] *
                # self.transitions[unobserved_dict[unobserved],
                #                  unobserved_dict[prev_unobserved]]
                # print(self.t1)
                input()

        return 1
