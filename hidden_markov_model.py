# import necessary modules
import numpy as np
import pandas as pd


class HiddenMarkovModel():

    def __init__(self, length):
        self.length = length

    def viterbi(self):
        print('in viterbi method')
