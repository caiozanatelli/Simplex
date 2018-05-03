import numpy as np
from fractions import Fraction

class LinearProgramming:
    __tableau  = None
    __num_rows = 0
    __num_rows = 0   
 
    def __init__(self, num_rows, num_cols, input_matrix):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__make_tableau(input_matrix)
        #print(self.get_tableau())

    def get_tableau(self):
        return self.__tableau

    def __make_tableau(self, input_matrix):
        c = input_matrix[0][:-1]
        A = input_matrix[1:,:-1]
        b = np.matrix(input_matrix[1:, -1])
        aux_vars_fpi = np.identity(self.__num_rows)
        hist_op = np.concatenate((np.zeros((1, self.__num_rows)), np.identity(self.__num_rows)), axis=0)

        print("c:")
        print(c)
        print("\nA:")
        print(A)
        print("\nb:")
        print(b.T)
        print("\nAux Variables for FPI")
        print(aux_vars_fpi)
        print("\nHistory Operations Matrix:")
        print(hist_op)
        print("")
        
        self.__tableau = np.column_stack((np.column_stack((hist_op,
                            np.row_stack((-1*c, A)))),
                            np.concatenate((np.zeros((1, 1)), b.T), axis=0)))

    
        #print(self.__tableau)
        for i in xrange(0, self.__tableau.shape[0]):
            for j in xrange(0, self.__tableau.shape[1]):
                self.__tableau[i, j] = Fraction(self.__tableau[i, j])
        
        print("Tableau successfully created!")
        print("")
        print(self.__tableau)
        print("")

