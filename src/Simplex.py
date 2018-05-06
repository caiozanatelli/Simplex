import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from IOUtils import IOUtils
from copy import deepcopy

class Simplex:
    LP_INFEASIBLE         = 0
    LP_FEASIBLE_UNBOUNDED = 1
    LP_FEASIBLE_BOUNDED   = 2

    __lp = None
    __obj_func_c = None
    __feasible_base_solution = None


    def __init__(self, num_rows, num_cols, input_matrix):
        IOUtils.print_header_line_screen()
        print(">> Starting Simplex...")
        IOUtils.print_header_line_screen()

        # Create the tableau that represents the linear programming
        self.__lp = LinearProgramming(2, 3, input_matrix)
        # Set the c objective function and the b array
        self.__obj_func_c = input_matrix[0][:-1]


    def solve(self):
        IOUtils.print_header_line_screen()
        # Get the neg entries in both c and b arrays from the tableau
        index_neg_entry_in_c   = self.__lp.get_first_neg_entry_col_in_c()
        index_list_b_neg_values = self.__lp.get_b_neg_entries_rows()

        # The c array is in optimum state. Dual Simplex must be used
        if index_neg_entry_in_c < 0:
            print(">> The c array is in optimum status: Dual Simplex will be used.")
            self.__apply_dual_simplex()
        else:
            print(">> The c array is not in optimum status: Primal Simplex will be used.")
            
            if not index_list_b_neg_values:
                print(">>>> All entries in b are non-negative. No need for an auxiliar LP.")
                self.__apply_primal_simplex() # Solve LP through Primal Simplex algorithm
            else:
                print(">>>> There are negative entries in b. An auxiliar LP is needed.")
                print(">>>> Creating auxiliar LP to find a feasible basic solution to the problem")

                # Create an auxiliar LP to find a basic solution to the original problem
                simplex_aux = deepcopy(self)
                simplex_aux.__lp.make_auxiliar_lp(index_list_b_neg_values)
                
                # Pivotate the extra variables to prepare the auxiliar tableau for the primal simplex
                rows = simplex_aux.__lp.get_tableau_num_rows()
                cols = simplex_aux.__lp.get_tableau_num_cols()
                for i in range(1, rows):
                    simplex_aux.__pivotate_element(i, cols - rows + i - 1)

                primal_simplex_result = simplex_aux.__apply_primal_simplex()

                # Primal Simplex for auxiliar LP is always feasible
                if (primal_simplex_result[1] == 0):
                    # Pivotate the base columns found in the auxiliar LP
                    base_columns = primal_simplex_result[2]
                    for i in range(1, rows):
                        self.__pivotate_element(i, base_columns[i])
                    
                    self.__apply_primal_simplex()
                    #print(self.__lp.get_tableau())

                # TODO: use the auxiliar solution to solve the original linear programming

    def __pivotate_element(self, row, col):
        """
        Pivotate the element at position (row, col) in the tableau.
        """
        print(">>>> Pivotating element at position (" + str(row) + ", " + str(col) + ")")

        tableau_num_rows = self.__lp.get_tableau_num_rows()
        tableau_num_cols = self.__lp.get_tableau_num_cols()
        pivot_value = self.__lp.get_tableau_elem(row, col)

        if pivot_value == 0:
            return

        # Dividing the pivot line by the pivot's value
        if (pivot_value != 1):
            for i in xrange(tableau_num_cols):
                curr_elem = self.__lp.get_tableau_elem(row, i)
                self.__lp.set_tableau_elem(row, i, curr_elem / pivot_value)

            # This should result in 1
            pivot_value = self.__lp.get_tableau_elem(row, col)
    
        for i in xrange(tableau_num_rows):
            if i == row:
                continue
            if self.__lp.get_tableau_elem(i, col) == 0:
                continue

            sum_pivot_factor = self.__lp.get_tableau_elem(i, col)

            for j in xrange(tableau_num_cols):
                curr_elem = self.__lp.get_tableau_elem(i, j)
                elem_in_pivot_row = self.__lp.get_tableau_elem(row, j)
                new_elem_value = curr_elem - sum_pivot_factor*elem_in_pivot_row
                self.__lp.set_tableau_elem(i, j, new_elem_value)

        print(">>>>>> DONE.")
        print(">>>> Tableau after the pivotation: ")
        IOUtils.print_header_line_screen()
        print(self.__lp.get_tableau())
        IOUtils.print_header_line_screen()


    def __apply_primal_simplex(self):
        """
        Implementation of the Primal Simplex algorithm.

        Return whether the LP is feasible and bounded, feasible and unbounded, or
        infeasible. The certificates, the objective values as well as the feasible
        base columns are returned in a tuple, when applicable.
        """
        IOUtils.print_header_line_screen()
        print(">> Starting Primal Simplex")
        IOUtils.print_header_line_screen()

        num_rows = self.__lp.get_tableau_num_rows()
        num_cols = self.__lp.get_tableau_num_cols()
        is_lp_bounded = True

        feasible_base_columns = []
        feasible_base_columns.append(-1) # There's no pivot in the c line
        # Store the initial columns that make the basic solution
        for i in range(1, num_rows):
            feasible_base_columns.append(num_cols - num_rows - 1 + i)

        print(feasible_base_columns)

        while True:
            # Get neg entries in the a
            neg_entry_index_in_c = self.__lp.get_first_neg_entry_col_in_c()

            # There is no neg entry in c, then Primal Simplex is over
            if (neg_entry_index_in_c < 0):
                print(">>>> There is no entry in c to optimize. Simplex is over.")
                IOUtils.print_header_line_screen()

                is_lp_bounded = True
                break

            print(">>>> Searching for element to pivotate")
            # Choose one element (row, column) in the neg column to pivotate
            row, col = self.__get_primal_pivot_element(neg_entry_index_in_c)

            # There is no negative entry in the c array. The LP is then ilimited 
            if row < 0:
                print(">>>> There is no more elements to pivotate.")
                print(">>>> The LP is unbounded! <<<<")

                is_lp_bounded = False
                break
            else: # The element to be pivotated has been chosen
                print(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")

                self.__pivotate_element(row, col) # Pivotate the chosen element
                feasible_base_columns[row] = col  # Update the base columns to the basic feasible solution
        

        if is_lp_bounded:
            obj_value = self.__lp.get_objective_value()
            optimality_certificate = self.__lp.get_optimality_certificate()

            print(">> Maximum objective value: " + str(obj_value))
            print(">> Optimality certificate: " + str(optimality_certificate))
            IOUtils.print_header_line_screen()

            # Return the id for a feasible bounded solution and the base columns associated
            return (self.LP_FEASIBLE_BOUNDED, obj_value, feasible_base_columns, optimality_certificate)
        else:
            return (self.LP_FEASIBLE_UNBOUNDED)


    def __get_primal_pivot_element(self, neg_entry_index):
        """
        Select the element for Primal Simplex pivotation by choosing the one with minimum ratio.
        Return a pair (row, col) for the chosen element.
        """

        # Choosing the first neg column in the c array for pivotating
        pivot_col = neg_entry_index
        pivot_row = -1
        min_ratio = Fraction(-1)
        rows = self.__lp.get_tableau_num_rows()
        cols = self.__lp.get_tableau_num_cols()

        # Find the element with min ratio in the column pivot_col in matrix A for pivotating
        for i in xrange(rows - 1):
            b_entry_value = self.__lp.get_tableau_elem(i + 1, cols -1)
            tableau_candidate_pivot = self.__lp.get_tableau_elem(i + 1, pivot_col)

            # Check if the current element can be a candidate for pivot
            if tableau_candidate_pivot > 0:
                curr_ratio = b_entry_value / tableau_candidate_pivot

                # Check if we found an element with a better ratio
                if min_ratio < 0 or curr_ratio < min_ratio:
                    min_ratio = curr_ratio
                    pivot_row = i + 1

        # If pivot_row = -1, then the LP is ilimited
        return pivot_row, pivot_col


    def __apply_dual_simplex(self):
        pass


    def __get_dual_pivot_element(self, neg_entry_index):
        """
        Select the element for Dual Simple pivotation by choosing the one with minimum ratio.
        Return a pair (row, col) for the chosen element.
        """

        pivot_row = neg_entry_index
        pivot_col = -1
        min_ratio = Fraction(-1)

        tableau_num_rows = self.__lp.get_tableau_num_rows()
        tableau_num_cols = self.__lp.get_tableau_num_cols()

        b_entry_value = self.__lp.get.tableau.elem(pivot_row, tableau_num_rows - 1)

        # Find the element with min ratio in the row pivote_row in matrix A for pivotating
        for i in xrange(tableau_num_cols - 1):
            tableau_candidate_pivot = self.__lp.get_tableau_elem(row, i)

            # Check if the current element can be a candidate for pivot
            if tableau_candidate_pivot < 0:
                curr_ratio = b_entry_value / tableau_candidate_pivot

                # Check if we found an element with a better ratio
                if min_ratio < 0 or curr_ratio < min_ratio:
                    min_ratio = curr_ratio
                    pivot_col = i

        # If pivot_col = -1, then there is no element to pivotate
        return pivot_row, pivot_col
