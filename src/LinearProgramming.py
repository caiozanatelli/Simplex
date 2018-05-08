import numpy as np
from fractions import Fraction
import logging
import sys

class LinearProgramming:
    """
    Class for the linear programming representation. The internal structure
    is stored as Numpy matrixes whose elements are represented as fractions
    in order to get more precision. The LP representation is created by
    concatenating all the information given by input matrix, such as the
    A, b, and c arrays.

    Note that FPI representation is also implemented. 
    """
    __tableau     = None
    __num_rows    = 0
    __num_cols    = 0
    __lp_init_col = 0   
 

    def __init__(self, num_rows, num_cols, input_matrix):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__lp_init_col = num_rows
        # Create the tableau
        self.__make_tableau(input_matrix)
        # Put tableau in canonical extended form
        self.__tableau = self.get_extended_canonical_tableau()


    def get_extended_canonical_tableau(self):
        logging.debug(">>>> Generating extended canonical tableau...")
        logging.debug(">>>>>> DONE.")

        orig_tableau = np.copy(self.__tableau)
        num_rows = self.get_tableau_num_rows()
        num_cols = self.get_tableau_num_cols()
        id_matrix = np.identity(num_rows - 1)
        extended_lp  = np.zeros((num_rows, num_rows + num_cols - 1)).astype('object')
        
        for i in xrange(num_rows):
            # Put the tableau elements except the b array
            for j in xrange(num_cols - 1):
                extended_lp[i, j] = orig_tableau[i, j]
            # Put the identity matrix for fpi variables and add 0s to the c array
            for j in range(0, num_rows - 1):
                if i == 0:
                    extended_lp[i, j + num_cols - 1] = Fraction(0)
                else:
                    extended_lp[i, j + num_cols - 1] = Fraction(id_matrix[i - 1, j])
        # Fill the b array in the new tableau
        extended_lp[:, -1] = orig_tableau[:, -1]

        return extended_lp


    def make_auxiliar_lp(self, index_list_b_neg_values):
        # Multiplying lines by -1 in order to make the b array non-negative
        for i in index_list_b_neg_values:
            for j in xrange(0, self.get_tableau_num_cols()):
                elem = self.get_tableau_elem(i, j)
                self.set_tableau_elem(i, j, elem * (-1)) 

        # New tableau with extra variables
        self.set_tableau(self.get_extended_canonical_tableau())

        rows = self.get_tableau_num_rows()
        cols = self.get_tableau_num_cols()

        # Changing the objective function to the new variables
        for i in xrange(cols):
            value = 0
            if i < cols - 1 and i >= cols - rows:
                value = 1
            self.__tableau[0, i] = Fraction(value)


    def __make_tableau(self, input_matrix):
        """
        Create the extended tableau for representing the linear programming.
        This method also puts the LP in FPI.
        """
        logging.debug(">>>> Creating tableau for the linear programming...")
        logging.debug(">>>>>> DONE.")

        rows = input_matrix.shape[0]
        cols = input_matrix.shape[1] + rows - 1
        tableau = np.zeros((rows, cols)).astype('object')

        # Setting operation matrix in the tableau
        tableau[0, 0:rows-1]  = np.zeros((1, rows - 1))
        tableau[1:, 0:rows-1] = np.identity(rows - 1)
        tableau[:, cols - 1] = input_matrix[:, -1]
        tableau[:, rows-1:-1] = input_matrix[:, :-1]
        tableau[0, :-1] = -tableau[0, :-1]

        logging.debug(">>>> Setting Tableau elements as fractions...")
        logging.debug(">>>>>> DONE.")
        
        self.__tableau = tableau
        # Changing the tableau elements into fractions for better precision
        for i in xrange(0, self.__tableau.shape[0]):
            for j in xrange(0, self.__tableau.shape[1]):
                self.__tableau[i, j] = Fraction(self.__tableau[i, j])


    def set_tableau(self, tableau):
        self.__tableau = tableau


    def get_tableau(self):
        """
        Return the linear programming tableau
        """
        return self.__tableau


    def print_tableau(self):
        tableau_str = ""
        for i in xrange(self.__tableau.shape[0]):
            for j in xrange(self.__tableau.shape[1]):
                elem = round(self.__tableau[i, j], 4)
                tableau_str = tableau_str + str(elem) + "  "
            tableau_str = tableau_str + "\n"

        return tableau_str


    def get_LP_init_column(self):
        """
        Return the column index where the LP starts in the tableau.
        
        This is necessary because we added a matrix to left of the
        LP matrix for registering the Gaussian operations that have
        been applied.
        """
        return self.__lp_init_col


    def get_tableau_elem(self, i, j):
        """
        Return the element at position (i, j) in the tableau
        """
        return self.__tableau[i, j]


    def set_tableau_elem(self, i, j, value):
        """
        Set a value to the element at position (i, j) in the tableau
        """
        self.__tableau[i, j] = value


    def get_objective_value(self):
        """
        Return the objective value from the tableau. 

        Note that this value is optimum if the LP is limited and the 
        Simplex algorithm has been applied to the tableau. 
        """
        return self.__tableau[0, -1]


    def get_optimality_certificate(self):
        """
        Return the optimality certificate for the LP (Primal Simplex). 

        This certificate is obtained from the matrix that register the 
        operations over the tableau.
        """
        return self.__tableau[0, 0:self.__tableau.shape[0] - 1]


    def get_first_neg_entry_col_in_c(self):
        """
        Verify whether the 'c' array in the LP is in optimum status.
        
        Return the first index where there is a neg entry in c.
        Return -1 if there is none.
        """
        #print(self.__lp_init_col)
        #exit(0)
        for i in xrange(self.__lp_init_col, self.__tableau.shape[1] - 1):
            if (self.__tableau[0, i] < 0):
                return i

        return -1


    def get_first_neg_entry_row_in_b(self):
        """
        Verify whether there is at least one negative value in the 'b' array.
        
        Return the first index where there is a neg entry in b.
        Return -1 if there is none.
        """
        b_col = self.get_tableau_num_cols() - 1
        for i in xrange(1, self.__tableau.shape[0]):
            if self.__tableau[i, b_col] < 0:
                return i

        return -1


    def get_b_neg_entries_rows(self):
        """
        Verify whether the 'b' array in the LP has neg entries
        Return a list with all indexes where there is a neg entry in b.
        """
        rows_neg_entries_in_b = []
        for i in xrange(self.__tableau.shape[0] - 1):
            if (self.__tableau[i + 1, -1] < 0):
                rows_neg_entries_in_b.append(i + 1)

        return rows_neg_entries_in_b


    def get_tableau_num_rows(self):
        """
        Return the tableau's number of rows
        """
        return self.__tableau.shape[0]


    def get_tableau_num_cols(self):
        """
        Return the tableau's number of columns
        """
        return self.__tableau.shape[1]

