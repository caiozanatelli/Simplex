import numpy as np
from fractions import Fraction

class LinearProgramming:
    """
    Class for the linear programming representation. The internal structure
    is stored as Numpy matrixes whose elements are represented as fractions
    in order to get more precision. The LP representation is created by
    concatenating all the information given by input matrix, such as the
    A, b, and c arrays.

    Note that FPI representation is also implemented. 
    """
    __tableau  = None
    __num_rows = 0
    __num_rows = 0   
 
    def __init__(self, num_rows, num_cols, input_matrix):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__make_tableau(input_matrix)

    def get_tableau(self):
        """
        Return the linear programming tableau
        """
        return self.__tableau

    def __make_tableau(self, input_matrix):
        """
        Create the extended tableau for representing the linear programming.
        This method also puts the LP in FPI.
        """
        print(">> Creating canonical Tableau...")
    
        # Extracting components information from the linear programming
        c = input_matrix[0][:-1]
        A = input_matrix[1:,:-1]
        b = np.matrix(input_matrix[1:, -1])
        aux_vars_fpi = np.identity(self.__num_rows)
        hist_op = np.concatenate((np.zeros((1, self.__num_rows)), np.identity(self.__num_rows)), axis=0)
        
        # Setting tableau by concatenating all the arrays involved
        self.__tableau = np.column_stack((np.column_stack((hist_op,
                            np.row_stack((-1*c, A)))),
                            np.concatenate((np.zeros((1, 1)), b.T), axis=0)))

        print(">>>> Setting Tableau elements as fractions...")
        print(">>>> DONE.")
        
        # Changing the tableau elements into fractions for better precision
        for i in xrange(0, self.__tableau.shape[0]):
            for j in xrange(0, self.__tableau.shape[1]):
                self.__tableau[i, j] = Fraction(self.__tableau[i, j])
        
        print(">> DONE. The following Tableau has been created: ")
        print("*************************************************")
        print(self.get_tableau())
        print("*************************************************")
        print("")

