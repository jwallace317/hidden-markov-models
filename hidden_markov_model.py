# import necessary modules
import numpy as np


# hidden markov model class
class HiddenMarkovModel():

    # initialize hidden markov model
    def __init__(self, observations, states, emissions, transitions, length):
        self.observations = observations  # observation space
        self.states = states  # state space
        self.emissions = emissions  # emission matrix
        self.transitions = transitions  # transmission matrix
        self.length = length  # length of hmm
        self.t1 = np.zeros((len(states) - 1, length))
        self.t2 = np.zeros((len(states) - 1, length))

    def viterbi(self, sequence):

        return 1
