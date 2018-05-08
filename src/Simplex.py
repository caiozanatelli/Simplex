import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from IOUtils import IOUtils
from copy import deepcopy
import logging

class Simplex:
    LP_INFEASIBLE         = 0
    LP_FEASIBLE_UNBOUNDED = 1
    LP_FEASIBLE_BOUNDED   = 2

    __lp = None
    __obj_func_c = None
    __feasible_base_solution = None
    __io_utils = None


    def __init__(self, num_rows, num_cols, input_matrix, io_utils):
        logging.debug(IOUtils.print_header_line_screen())
        logging.debug(">> Starting Simplex.")
        logging.debug(IOUtils.print_header_line_screen())

        self.__io_utils = io_utils
        # Create the tableau that represents the linear programming
        self.__lp = LinearProgramming(num_rows, num_cols, input_matrix)
        # Set the c objective function and the b array
        self.__obj_func_c = input_matrix[0][:-1]


    def solve(self):
        """
        Main method to solve a linear programming problem. All the possible situations are
        evaluated in order to decide which Simplex algorithm to use (Primal or Dual).

        All operations are stored in a log file.
        """

        IOUtils.print_header_line_screen()
        # Get the neg entries in both c and b arrays from the tableau
        index_neg_entry_in_c   = self.__lp.get_first_neg_entry_col_in_c()
        index_list_b_neg_values = self.__lp.get_b_neg_entries_rows()
        
        # The c array is in optimum state. Dual Simplex must be used
        if index_neg_entry_in_c < 0:
            logging.debug(">> The c array is in optimum status: Dual Simplex will be used.")
            
            status, certificate, obj_value, solution = self.__apply_dual_simplex()
            message = str(status) + "\n"

            if status == self.LP_FEASIBLE_BOUNDED:
                message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)
                
                # Log on the screen
                print("##############################     \n" +
                      " . : Feasible and Bounded : .      \n " +
                      "##############################     \n" +
                      "[+] Solution: " + str(solution) + "\n" +
                      "[+] Objective Value: " + str(obj_value) + "\n" +
                      "[+] Optimality Certificate: " + str(certificate) + "\n")

            elif status == self.LP_INFEASIBLE:
                message = message + str(certificate)

                # Log on the screen
                print("[+] . : Infeasible : .\n " +
                      "[+] Infeasibility Certificate: " + str(certificate) + "\n")
            else:
                message = ">>>> Something terrible happened. Dual Simplex should never lead to unbounded!"
                logging.debug(message)
                print(message)

            self.__io_utils.write_output(message)
        else:
            logging.debug(">> The c array is not in optimum status: Primal Simplex will be used.")
            
            feasible_base_columns = []
            feasible_base_columns.append(-1) # There's no pivot in the c row
            rows = self.__lp.get_tableau_num_rows()
            cols = self.__lp.get_tableau_num_cols()

            if not index_list_b_neg_values:
                logging.debug(">>>> All entries in b are non-negative. No need for an auxiliar LP.")
                
                # Store the initial columns that make the basic solution
                for i in range(1, rows):
                    feasible_base_columns.append(cols - rows - 1 + i)

                # Apply Primal Simplex and get the result
                status,certificate,obj_value,feasible_base,solution = self.__apply_primal_simplex(feasible_base_columns)                
                # Store the string to print to the output file
                message = str(status) + "\n"
                if status == self.LP_FEASIBLE_BOUNDED:
                    message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)

                    # Log on the screen
                    print("############################## \n" + 
                          " . : Feasible and Bounded : .\n " +
                          "############################## \n" +
                          "[+] Solution: " + str(solution) + "\n" +
                          "[+] Objective Value: " + str(obj_value) + "\n" +
                          "[+] Optimality Certificate: " + str(certificate) + "\n")

                elif status == self.LP_FEASIBLE_UNBOUNDED:
                    message = message + str(certificate)
                    # Log on the screen
                    print("################################ \n" +
                          " . : Feasible and Unbounded : .  \n " +
                          "################################ \n" +
                          "[+] Unbounded Certificate: " + str(certificate) + "\n")

                else:
                    message = ">>>> Something terrible happened. The objective value is negative, and that is such an heresy!"
                    loggin.debug(message)
                    print(message)

                # Store the final result in the output file
                self.__io_utils.write_output(message)
            else:
                logging.debug(">>>> There are negative entries in b. An auxiliar LP is needed.")
                logging.debug(">>>> Creating auxiliar LP to find a feasible basic solution to the problem")

                # Create an auxiliar LP to find a basic solution to the original problem
                simplex_aux = deepcopy(self)
                simplex_aux.__lp.make_auxiliar_lp(index_list_b_neg_values)
                
                # Pivotate the extra variables to prepare the auxiliar tableau for the primal simplex
                rows = simplex_aux.__lp.get_tableau_num_rows()
                cols = simplex_aux.__lp.get_tableau_num_cols()

                # Store the initial columns that make the basic solution
                for i in range(1, rows):
                    simplex_aux.__pivotate_element(i, cols - rows + i - 1)
                    feasible_base_columns.append(cols - rows - 1 + i)

                # Primal Simplex for auxiliar LP is always feasible
                status,certificate,aux_obj_value,base_columns,sol = simplex_aux.__apply_primal_simplex(feasible_base_columns)
                if aux_obj_value == 0:
                    # Pivotate the base columns found in the auxiliar LP
                    for i in range(1, rows):
                        self.__pivotate_element(i, base_columns[i])
                    
                    # Apply the Primal Simplex Algorithm and get the result
                    status,certificate,obj_value,feasible_base,solution = self.__apply_primal_simplex(base_columns)
                    message = str(status) + "\n"
                    if status == self.LP_FEASIBLE_BOUNDED:
                        message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)

                        # Log on the screen
                        print("###############################    \n" + 
                              " . : Feasible and Bounded : .      \n " +
                              "###############################    \n" + 
                              "[+] Solution: " + str(solution) + "\n" +
                              "[+] Objective Value: " + str(obj_value) + "\n" +
                              "[+] Optimality Certificate: " + str(certificate) + "\n")
                    
                    elif status == self.LP_FEASIBLE_UNBOUNDED:
                        message = message + str(certificate)

                        # Log on the screen
                        print("############################### \n" + 
                              " . : Feasible and Unounded : .  \n" +
                              "############################### \n" +
                              "[+] Unbounded Certificate: " + str(certificate) + "\n")

                    # Print the result to the output file
                    self.__io_utils.write_output(message)
                    
                elif aux_obj_value < 0:
                    logging.debug(">>>> The original LP is infeasible.")
                    logging.debug(">>>> Infeasibility Certificate: " + str(certificate))

                    message = str(self.LP_INFEASIBLE) + "\n" + str(certificate)
                    self.__io_utils.write_output(message)
                    
                    # Log on the screen
                    print("#################### \n" +
                          " . : Infeasible : .  \n " +
                          "#################### \n" + 
                          "[+] Infeasibility Certificate: " + str(certificate) + "\n")
                else:
                    message = ">>>> Something terrible happened. The objective value is negative, and that is such an heresy!"
                    logging.debug(message)
                    print(message)


    def __pivotate_element(self, row, col):
        """
        Pivotate the element at position (row, col) in the tableau.
        """
        logging.debug(">>>> Pivotating element at position (" + str(row) + ", " + str(col) + ")")

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

        logging.debug(">>>>>> DONE.")
        logging.debug(">>>> Tableau after the pivotation: ")
        logging.debug(IOUtils.print_header_line_screen())
        logging.debug(self.__lp.print_tableau())
        logging.debug(IOUtils.print_header_line_screen())


    def __apply_primal_simplex(self, feasible_base_columns):
        """
        Implementation of the Primal Simplex algorithm.

        Return whether the LP is feasible and bounded, feasible and unbounded, or
        infeasible. The certificates, the objective values as well as the feasible
        base columns are returned in a tuple, when applicable.
        """
        logging.debug(IOUtils.print_header_line_screen())
        logging.debug(">> Starting Primal Simplex")
        logging.debug(IOUtils.print_header_line_screen())

        num_rows = self.__lp.get_tableau_num_rows()
        num_cols = self.__lp.get_tableau_num_cols()
        is_lp_bounded = True

        while True:
            # Get neg entries in the a
            neg_entry_index_in_c = self.__lp.get_first_neg_entry_col_in_c()

            # There is no neg entry in c, then Primal Simplex is over
            if (neg_entry_index_in_c < 0):
                logging.debug(">>>> There is no entry in c to optimize. Primal Simplex is over.")
                logging.debug(IOUtils.print_header_line_screen())

                is_lp_bounded = True
                break

            logging.debug(">>>> Searching for element to pivotate...")
            # Choose one element (row, col) in the neg column to pivotate
            row, col = self.__get_primal_pivot_element(neg_entry_index_in_c)

            # There is no negative entry in the c array. The LP is then ilimited 
            if row < 0:
                logging.debug(">>>> There is no more elements to pivotate.")
                logging.debug(">>>> The LP is unbounded! <<<<")
                is_lp_bounded = False
                break
            else: # The element to be pivotated has been chosen
                logging.debug(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")

                self.__pivotate_element(row, col) # Pivotate the chosen element
                feasible_base_columns[row] = col  # Update the base columns to the basic feasible solution
        
        # Check whether the LP is bounded
        if is_lp_bounded:
            obj_value = round(self.__lp.get_objective_value(), 6)
            optimality_certificate = list(self.__lp.get_optimality_certificate())

            num_vars = self.__lp.get_tableau_num_cols() - 2*self.__lp.get_tableau_num_rows() + 1
            solution = [0]*num_vars # Starting solution with zeros
            for i in xrange(self.__lp.get_LP_init_column(), num_cols - num_rows):
                # If this column belongs to the base, then we add b solution to the variable its variable
                if i in feasible_base_columns:
                    row_sol_base = feasible_base_columns.index(i)
                    solution[i - num_rows + 1] = self.__lp.get_tableau_elem(row_sol_base, num_cols - 1)

            # Turn the values into float
            for i in xrange(num_vars): 
                solution[i] = round(solution[i], 6)
            for i in xrange(len(optimality_certificate)):
                optimality_certificate[i] = round(optimality_certificate[i], 6)

            logging.debug(IOUtils.print_header_line_screen())
            logging.debug(">> Maximum objective value: " + str(obj_value))
            logging.debug(">> Solution: "                + str(solution))
            logging.debug(">> Optimality certificate: "  + str(optimality_certificate))
            logging.debug(IOUtils.print_header_line_screen())

            # Return the id for a feasible bounded solution and the base columns associated
            return (self.LP_FEASIBLE_BOUNDED, optimality_certificate, obj_value, feasible_base_columns, solution)
        else:
            # TODO: Fix the unbounded certificate
            num_vars = num_cols - num_rows
            unbounded_certificate = [0]*num_vars
            unbounded_certificate[col - num_rows + 1] = 1

            for i in xrange(self.__lp.get_LP_init_column(), num_cols - 1):
                if i in feasible_base_columns:
                    cert_row = feasible_base_columns.index(i)
                    neg_column_elem = -self.__lp.get_tableau_elem(cert_row, col)
                    unbounded_certificate[i - num_rows + 1] = float(neg_column_elem)
            
            for i in xrange(num_vars): # Turn the values into float
                unbounded_certificate[i] = round(unbounded_certificate[i], 6)
            
            logging.debug(">>>> Unbounded certificate: " + str(unbounded_certificate))
            return (self.LP_FEASIBLE_UNBOUNDED, unbounded_certificate, 0, None, None)


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
        logging.debug(IOUtils.print_header_line_screen())
        logging.debug(">> Starting Dual Simplex")
        logging.debug(IOUtils.print_header_line_screen())

        num_rows = self.__lp.get_tableau_num_rows()
        num_cols = self.__lp.get_tableau_num_cols()
        is_lp_feasible = True

        feasible_base_columns = [-1]*num_rows

        while True:
            neg_entry_in_b = self.__lp.get_first_neg_entry_row_in_b()

            # There is no neg entry in b, then Dual Simplex is over
            if neg_entry_in_b < 0:
                logging.debug(">>>> There is no negative entry in b. Dual Simplex is over.")
                logging.debug(IOUtils.print_header_line_screen())

                is_lp_feasible = True
                break

            logging.debug(">>>> Searching for element to pivotate...")
            # Choose an element (row, col) in the neg row of b to pivotate
            row, col = self.__get_dual_pivot_element(neg_entry_in_b)

            if col < 0:
                logging.debug(">>>> There is no more elements to pivotate.")
                logging.debug(">>>> The LP is infeasible! <<<<")
                is_lp_feasible = False
            else:
                logging.debug(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")

                self.__pivotate_element(row, col) # Pivotate the chosen element
                feasible_base_columns[row] = col

        # Check whether the LP is feasible
        if is_lp_feasible:
            obj_value = round(self.__lp.get_objective_value(), 6)
            optimality_certificate = list(self.__lp.get_optimality_certificate())

            num_vars = num_cols - 2*num_rows + 1
            solution = [0]*num_vars # Starting solution with zeros
            for i in xrange(self.__lp.get_LP_init_column(), num_cols - num_rows):
                # If this column belongs to the base, then we add b solution to the variable its variable
                if i in feasible_base_columns:
                    row_sol_base = feasible_base_columns.index(i)
                    solution[i - num_rows + 1] = self.__lp.get_tableau_elem(row_sol_base, num_cols - 1)

            # Turn the values into float
            for i in xrange(num_vars): 
                solution[i] = round(solution[i], 6)
            for i in xrange(len(optimality_certificate)):
                optimality_certificate[i] = round(optimality_certificate[i], 6)

            logging.debug(IOUtils.print_header_line_screen())
            logging.debug(">> Maximum objective value: " + str(obj_value))
            logging.debug(">> Solution: "                + str(solution))
            logging.debug(">> Optimality certificate: "  + str(optimality_certificate))
            logging.debug(IOUtils.print_header_line_screen())

            return (self.LP_FEASIBLE_BOUNDED, optimality_certificate, obj_value, solution)
        else:
            # Get the infeasibility certificate from the operation matrix
            infeasible_certificate = self.__lp.get_tableau()[row, 0:num_rows - 1]
            # Turn the values into float
            for i in xrange(len(infeasible_certificate)):
                infeasible_certificate[i] = round(infeasible_certificate[i], 6)

            return (self.LP_INFEASIBLE, infeasible_certificate, 0, None)


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
