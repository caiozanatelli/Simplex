import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming

class Simplex:
    __lp = None
    __obj_func_c = None
    __b_array = None

    def __init__(self, num_rows, num_cols, input_matrix):
        print(">> Starting Simplex...")
        
        # Create the tableau that represents the linear programming
        self.__lp = LinearProgramming(2, 3, input_matrix)
        # Set the c objective function and the b array
        self.__obj_func_c = input_matrix[0][:-1]
        self.__update_curr_b_array()


    def solve(self):
        index_list_c_neg_values = self.__lp.get_c_neg_entries_cols()
        
        if not index_list_c_neg_values:
            print(">> The c array in the linear programming is in great status.")
        else:
            print(">> The c array in the linear programming is not in great status.")

            index_list_b_neg_values = self.__lp.get_b_neg_entries_rows()
            #if not index_list_b_neg_values:
            #    self.__primal_simplex(index_list_c_neg_values)
            #else:
            aux_lp = self.__create_auxiliar_lp()


    def __create_auxiliar_lp(self):
        print(">>>> Creating auxiliar linear programming tableau for finding a basic solution...")
        print(self.__lp.get_extended_canonical_tableau())


    def __update_curr_b_array(self):
        self.__b_array = self.__lp.get_tableau()[1:,-1]


    def __primal_simplex(self, index_list_c_neg_values):
        pivot_elem_index = self.__get_primal_pivot_element(index_list_c_neg_values)
        

    def __dual_simplex(self):
        pass


    def __get_primal_pivot_element(self, index_list_c_neg_values):
        # Choosing the first neg column in the c array for pivotating
        pivot_col = index_list_c_neg_values[0]
        pivot_row = -1
        min_ratio = Fraction(-1)

        # Find the element with min ratio in the column pivot_col in matrix A for pivotating
        for i in xrange(self.__lp.get_tableau().shape[0] - 1):
            tableau_candidate_pivot = self.__lp.get_tableau_elem(i + 1, pivot_col)

            # Check if we found an element with a better ratio
            if tableau_candidate_pivot > 0:
                curr_ratio = self.__b_array[i] / tableau_candidate_pivot
                if (curr_ratio > min_ratio):
                    pivot_row = i + 1

        # If pivot_row = -1, then the LP is ilimited
        return pivot_row, pivot_col
