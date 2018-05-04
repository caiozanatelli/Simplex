import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from IOUtils import IOUtils

class Simplex:
    __lp = None
    __obj_func_c = None
    __b_array = None
    __feasible_base_solution = None


    def __init__(self, num_rows, num_cols, input_matrix):
        IOUtils.print_header_line_screen()
        print(">> Starting Simplex...")
        IOUtils.print_header_line_screen()

        # Create the tableau that represents the linear programming
        self.__lp = LinearProgramming(2, 3, input_matrix)
        # Set the c objective function and the b array
        self.__obj_func_c = input_matrix[0][:-1]
        self.__update_curr_b_array()


    def solve(self):
        IOUtils.print_header_line_screen()

        # Get the neg entries in both c and b arrays from the tableau
        index_neg_entry_in_c   = self.__lp.get_first_neg_entry_col_in_c()
        index_list_b_neg_values = self.__lp.get_b_neg_entries_rows()

        if index_neg_entry_in_c < 0:
            print(">> The c array is in optimum status: Dual Simplex will be used.")
        else:
            print(">> The c array is not in optimum status: Primal Simplex will be used.")
            
            if not index_list_b_neg_values:
                print(">>>> All entries in b are non-negative. No need for an auxiliar LP.")
                # Solve LP through Primal Simplex algorithm
                self.__apply_primal_simplex()
            else:
                print(">>>> There are negative entries in b. An auxiliar LP is needed.")
                # Create an auxiliar LP to find a basic solution to the original problem
                aux_lp = self.__create_auxiliar_lp()


    def __create_auxiliar_lp(self):
        print(">>>> Creating auxiliar linear programming tableau for finding a basic solution...")
        print(self.__lp.get_extended_canonical_tableau())


    def __update_curr_b_array(self):
        self.__b_array = self.__lp.get_tableau()[1:,-1]


    def __apply_primal_simplex(self):
        IOUtils.print_header_line_screen()
        print(">> Starting Primal Simplex")
        IOUtils.print_header_line_screen()

        lp_has_unique_optimum_value = True

        while True:
            # Get neg entries in the a
            neg_entry_index_in_c = self.__lp.get_first_neg_entry_col_in_c()

            # There is no neg entry in c, then Primal Simplex is over
            if (neg_entry_index_in_c < 0):
                print(">>>> There is no entry in c to optimize. Simplex is over.")
                IOUtils.print_header_line_screen()

                lp_has_unique_optimum_value = True
                break

            print(">>>> Searching for element to pivotate")
            # Choose one element (row, column) in the neg column to pivotate
            row, col = self.__get_primal_pivot_element(neg_entry_index_in_c)

            # There is no negative entry in the c array. The LP is then ilimited 
            if row < 0:
                print(">>>> There is no more elements to pivotate.")
                print(">>>> The LP is ilimited! <<<<")

                lp_has_unique_optimum_value = False
                break
            else: # There is element to be pivotated has been chosen
                print(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")

                # Apply Primal Simplex on the tableau
                self.__primal_pivotation(row, col)
        

        if lp_has_unique_optimum_value:    
            print(">> Maximum objective value: " + str(self.__lp.get_objective_value()))
            print(">> Optimality certificate: " + str(self.__lp.get_optimality_certificate()))
            IOUtils.print_header_line_screen()
        else:
            pass


    def __apply_dual_simplex(self):
        pass


    def __get_primal_pivot_element(self, neg_entry_index):
        # Choosing the first neg column in the c array for pivotating
        pivot_col = neg_entry_index
        pivot_row = -1
        min_ratio = Fraction(-1)

        # Find the element with min ratio in the column pivot_col in matrix A for pivotating
        for i in xrange(self.__lp.get_tableau().shape[0] - 1):
            tableau_candidate_pivot = self.__lp.get_tableau_elem(i + 1, pivot_col)

            # Check if we found an element with a better ratio
            if tableau_candidate_pivot > 0:
                curr_ratio = self.__b_array[i] / tableau_candidate_pivot

                if min_ratio < 0 or curr_ratio < min_ratio:
                    min_ratio = curr_ratio
                    pivot_row = i + 1

        # If pivot_row = -1, then the LP is ilimited
        return pivot_row, pivot_col


    def __primal_pivotation(self, row, col):
        print(">>>> Pivotating element at position (" + str(row) + ", " + str(col) + ")")

        tableau_num_rows = self.__lp.get_tableau_num_rows()
        tableau_num_cols = self.__lp.get_tableau_num_cols()
        pivot_value = self.__lp.get_tableau_elem(row, col)

        # Dividing the pivot line by the pivot's value
        if (pivot_value != 1):
            for i in xrange(tableau_num_cols):
                curr_elem = self.__lp.get_tableau_elem(row, i)
                self.__lp.set_tableau_elem(row, i, curr_elem / pivot_value)

            # This should result in 1
            pivot_value = self.__lp.get_tableau_elem(row, col)
        else:
            for i in xrange(tableau_num_rows):
                if (i == row):
                    continue

                sum_pivot_factor  = -self.__lp.get_tableau_elem(i, col)

                for j in xrange(tableau_num_cols):
                    curr_elem = self.__lp.get_tableau_elem(i, j)
                    elem_in_pivot_row = self.__lp.get_tableau_elem(row, j)

                    new_elem_value = curr_elem + sum_pivot_factor*elem_in_pivot_row
                    self.__lp.set_tableau_elem(i, j, new_elem_value)


        print(">>>>>> DONE.")
        print(">>>> Tableau after the pivotation: ")
        IOUtils.print_header_line_screen()
        print(self.__lp.get_tableau())
        IOUtils.print_header_line_screen()