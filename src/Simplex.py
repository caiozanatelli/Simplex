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
                simplex_result = self.__apply_primal_simplex() # Solve LP through Primal Simplex algorithm
                base = simplex_result[2]

                num_vars = self.__lp.get_tableau_num_cols() - 2*self.__lp.get_tableau_num_rows() + 1
                solution = [0]*num_vars

                # Getting solution from base
                for i in xrange(1, len(base)):
                    r = base[i]
                    if r < self.__lp.get_tableau_num_cols() - self.__lp.get_tableau_num_rows():
                        solution[i] = self.__lp.get_tableau_elem(i, self.__lp.get_tableau_num_cols() - 1)

                print(solution)
                
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

                aux_simplex_result = simplex_aux.__apply_primal_simplex()
                aux_lp_obj_value = aux_simplex_result[1]
                
                # Primal Simplex for auxiliar LP is always feasible
                if aux_lp_obj_value == 0:
                    # Pivotate the base columns found in the auxiliar LP
                    base_columns = aux_simplex_result[2]
                    for i in range(1, rows):
                        self.__pivotate_element(i, base_columns[i])
                    self.__apply_primal_simplex()
                elif aux_lp_obj_value < 0:
                    print(">>>> The original LP is infeasible.")
                    print(">>>> Infeasible certificate: " + str(aux_simplex_result[3]))
                else:
                    print(">>>> Something terrible happened. The objective value is negative, and that is such an heresy!")


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
        feasible_base_columns.append(-1) # There's no pivot in the c row
        # Store the initial columns that make the basic solution
        for i in range(1, num_rows):
            feasible_base_columns.append(num_cols - num_rows - 1 + i)

        while True:
            # Get neg entries in the a
            neg_entry_index_in_c = self.__lp.get_first_neg_entry_col_in_c()

            # There is no neg entry in c, then Primal Simplex is over
            if (neg_entry_index_in_c < 0):
                print(">>>> There is no entry in c to optimize. Primal Simplex is over.")
                IOUtils.print_header_line_screen()

                is_lp_bounded = True
                break

            print(">>>> Searching for element to pivotate...")
            # Choose one element (row, col) in the neg column to pivotate
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
            print(">> Optimality certificate: "  + str(optimality_certificate))
            IOUtils.print_header_line_screen()

            # Return the id for a feasible bounded solution and the base columns associated
            return (self.LP_FEASIBLE_BOUNDED, obj_value, feasible_base_columns, optimality_certificate)
        else:
            unbounded_certificate = [0]*num_cols
            unbounded_certificate[col] = 1
            #for i in xrange(num_cols - num_rows + 1):
            
            for i in xrange(1, num_rows):
                col_base = feasible_base_columns[i]
                neg_column_elem = -self.__lp.get_tableau_elem(i, col)
                unbounded_certificate[col_base] = neg_column_elem
                #print(len(unbounded_certificate))

            print(unbounded_certificate)
            print(col)
            #print(col)
            #print(num_cols)
            #print(num_rows)
            #print(feasible_base_columns)
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
        for i in xrange(1, rows):
            b_entry_value = self.__lp.get_tableau_elem(i, cols -1)
            tableau_candidate_pivot = self.__lp.get_tableau_elem(i, pivot_col)

            # Check if the current element can be a candidate for pivot
            if tableau_candidate_pivot > 0:
                curr_ratio = b_entry_value / tableau_candidate_pivot

                # Check if we found an element with a better ratio
                if min_ratio < 0 or curr_ratio < min_ratio:
                    min_ratio = curr_ratio
                    pivot_row = i

        # If pivot_row = -1, then the LP is ilimited
        return pivot_row, pivot_col


    def __apply_dual_simplex(self):
        """
        Implementation of the Primal Simplex algorithm.

        Return whether the LP is feasible and bounded, feasible and unbounded, or
        infeasible. The certificates, the objective values as well as the feasible
        base columns are returned in a tuple, when applicable.
        """
        IOUtils.print_header_line_screen()
        print(">> Starting Dual Simplex")
        IOUtils.print_header_line_screen()

        num_rows = self.__lp.get_tableau_num_rows()
        num_cols = self.__lp.get_tableau_num_cols()
        is_lp_feasible = True

        while True:
            neg_entry_in_b = self.__lp.get_first_neg_entry_row_in_b()

            # There is no neg entry in b, then Dual Simplex is over
            if neg_entry_in_b < 0:
                print(">>>> There is no negative entry in b. Dual Simplex is over.")
                IOUtils.print_header_line_screen()

                is_lp_feasible = True
                break

            print(">>>> Searching for element to pivotate...")
            # Choose an element (row, col) in the neg row of b to pivotate
            row, col = self.__get_dual_pivot_element(neg_entry_in_b)

            if col < 0:
                print(">>>> There is no more elements to pivotate.")
                print(">>>> The LP is infeasible! <<<<")
                is_lp_feasible = False
            else:
                print(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")
                self.__pivotate_element(row, col) # Pivotate the chosen element

        if is_lp_feasible:
            obj_value = self.__lp.get_objective_value()
            optimality_certificate = self.__lp.get_optimality_certificate()

            print(">> Maximum objective value: " + str(obj_value))
            print(">> Optimality certificate: "  + str(optimality_certificate))
            IOUtils.print_header_line_screen()

            return self.LP_FEASIBLE_BOUNDED, obj_value, optimality_certificate
        else:
            infeasible_certificate = []
            return self.LP_INFEASIBLE


    def __get_dual_pivot_element(self, neg_entry_index):
        """
        Select the element for Dual Simplex pivotation by choosing the one with minimum ratio.
        Return a pair (row, col) for the chosen element.
        """
        pivot_row = neg_entry_index
        pivot_col = -1
        min_ratio = Fraction(-1)

        rows = self.__lp.get_tableau_num_rows()
        cols = self.__lp.get_tableau_num_cols()

        for i in range(rows -1, cols - 1):
            c_entry_value = self.__lp.get_tableau_elem(0, i)
            tableau_candidate_pivot = self.__lp.get_tableau_elem(pivot_row, i)

            if tableau_candidate_pivot < 0:
                curr_ratio = abs(c_entry_value / tableau_candidate_pivot)

                if min_ratio < 0 or curr_ratio < min_ratio:
                    min_ratio = curr_ratio
                    pivot_col = i

        # If pivot_col = -1, then there is no element to pivotate
        return pivot_row, pivot_col
