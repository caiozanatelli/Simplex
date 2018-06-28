import numpy as np
from fractions import Fraction
import logging
import sys
import math
from copy import deepcopy
from Utils import Utils

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
    #__num_rows_orig = 0
    #__num_cols_orig = 0 
 

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



    def add_restriction(self, new_row, coeficient=1):
        """
        Add a new restriction to the tableau. For this to happen, we must add the equation row
        and also insert a column in the operation matrix and another column in the A matrix, just
        before the b array. This method is used in the cutting plane algorithm in order to find
        an integer solution to the linear programming.
        """
        logging.debug(">>>> Adding a restriction to the tableau....")
        logging.debug(">>>>>> DONE.")

        rows = self.__tableau.shape[0] + 1
        cols = self.__tableau.shape[1] + 2
        tableau = np.zeros((rows, cols)).astype('object')

        # Setting operation matrix in the tableau
        tableau[0:rows-1, 0:rows-2] = self.__tableau[0:rows-1, 0:rows-2]
        tableau[rows-1, rows-2] = Fraction(1)

        # Setting the A matrix
        tableau[0:rows-1, rows-1:cols-2] = self.__tableau[0:rows-1, rows-2:cols-3]
        tableau[rows-1, rows-1:cols-2]   = new_row[rows-2:cols-3]
        tableau[rows-1, cols-2] = Fraction(coeficient)
        tableau[rows-1, cols-1] = new_row[cols-3]

        # Setting the b array
        tableau[0:rows-1, cols-1] = self.__tableau[0:rows-1, cols-3]


        for i in xrange(0, tableau.shape[0]):
            for j in xrange(0, tableau.shape[1]):
                tableau[i, j] = Fraction(tableau[i, j])

        self.__tableau = tableau
        self.__num_rows, self.__num_cols = tableau.shape
        self.__lp_init_col = tableau.shape[0] - 1


    def pivotate_element(self, row, col):
        """
        Pivotate the element at position (row, col) in the tableau.
        """
        logging.debug(">>>> Pivotating element at position (" + str(row) + ", " + str(col) + "): " + str(self.__tableau[row, col]))

        tableau_num_rows = self.get_tableau_num_rows()
        tableau_num_cols = self.get_tableau_num_cols()
        pivot_value = self.get_tableau_elem(row, col)

        if pivot_value == 0:
            return

        # Dividing the pivot line by the pivot's value
        if (pivot_value != 1):
            for i in xrange(tableau_num_cols):
                curr_elem = self.get_tableau_elem(row, i)
                self.set_tableau_elem(row, i, curr_elem / pivot_value)

            # This should result in 1
            pivot_value = self.get_tableau_elem(row, col)
    
        for i in xrange(tableau_num_rows):
            if i == row:
                continue
            if self.get_tableau_elem(i, col) == 0:
                continue

            sum_pivot_factor = self.get_tableau_elem(i, col)

            for j in xrange(tableau_num_cols):
                curr_elem = self.get_tableau_elem(i, j)
                elem_in_pivot_row = self.get_tableau_elem(row, j)
                new_elem_value = curr_elem - sum_pivot_factor*elem_in_pivot_row
                self.set_tableau_elem(i, j, new_elem_value)

        logging.debug(">>>>>> DONE.")
        logging.debug(">>>> Tableau after the pivotation: ")
        logging.debug(Utils.get_header_line_screen())
        logging.debug(self.print_tableau())
        logging.debug(Utils.get_header_line_screen())


    def get_floor_row(self, row):
        """
        Turn a row of the tableau into integer by applying a floor operation.
        """
        new_row = deepcopy(self.__tableau[row, :])
        for i in xrange(len(new_row)):
            new_row[i] = Fraction(math.floor(new_row[i]))

        return new_row


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


    def is_b_all_null(self):
        """
        Check whether the b array is all zero. In this case, the LP
        is feasible and bounded and the corresponding solution is the
        trivial one.
        """
        for i in xrange(1, self.__tableau.shape[0]):
            if self.__tableau[i, self.__tableau.shape[1] - 1] != 0:
                return False
        return True


    def is_any_row_inconsistent(self):
        """
        Check whether there is a row such that at least all variables are
        equal to zero and the b entry is different than zero. In this case,
        this equation is an absurd and hence there is no solution to the LP.
        """
        rows = self.__tableau.shape[0]
        cols = self.__tableau.shape[1]
        
        for i in xrange(1, rows):
            is_row_null = True
            for j in xrange(self.__lp_init_col, cols - 1):
                if self.__tableau[i, j] != 0:
                    is_row_null = False
                    break
            if is_row_null == True and self.__tableau[i, self.__tableau.shape[1] - 1] != 0:
                return i

        return -1


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


    def is_objective_value_integer(self):
        return self.get_objective_value().denominator == 1


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


    def is_column_in_original_vars(self, column):
        """
        Verify if a given column belongs to the problem's original variables.
        Return True if it does belong to the original variables and False otherwise.
        """
        num_rows = self.get_tableau_num_rows()
        num_cols = self.get_tableau_num_cols()

        return (column >= num_rows - 1 and column < num_cols - num_rows)


    def get_first_row_frac_in_b(self, feasible_basis, list_equations_used):
        """
        Return the first fraction element in b that corresponds to a variable in the feasible
        basis columns if, and only if, this column represents an original variable of the LP.
        If there is no such entry in b, the return is -1.
        """
        num_rows = self.get_tableau_num_rows()
        num_cols = self.get_tableau_num_cols()
        col_b    = num_cols - 1

        #for i in xrange(1, len(feasible_basis)):
        for i in xrange(1, num_rows):
        #    if i in list_equations_used:
        #        continue

            #col = feasible_basis[i]
            
            col = i
            #if self.is_column_in_original_vars(col):
            if self.__tableau[i, col_b].denominator != 1:
                list_equations_used.append(i)
                return i

        return -1


    def get_solution_from_feasible_basis(self, solution, feasible_basis):
        """
        Get the linear programming solution (in b array) through the feasible column basis found
        in the end of the Simplex algorithm.
        """
        num_rows    = self.get_tableau_num_rows()
        num_cols    = self.get_tableau_num_cols()
        lp_init_col = self.get_LP_init_column()
        num_vars    = num_cols - 2*num_rows + 1

        #solution    = [0]*num_vars # Starting solution with zeros

        for i in xrange(lp_init_col, num_cols - num_rows):
            # If this column belongs to the base, then we add b solution to the variable its variable
            if i in feasible_basis:
                row_sol_basis = feasible_basis.index(i)
                solution[i - num_rows + 1] = self.get_tableau_elem(row_sol_basis, num_cols - 1)

        # Turn the values into float
        #for i in xrange(num_vars): 
        #    solution[i] = round(solution[i], 6)

        return solution


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

