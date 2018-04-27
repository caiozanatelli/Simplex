import numpy as np
from fractions import Fraction

class LinearProgramming:
    
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def make_tableau(self, input_matrix):
        history_operations_matrix = np.zeros((1, self.num_rows)) + Fraction()
        
        
        print(history_operations_matrix)
