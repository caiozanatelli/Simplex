import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from IOUtils import IOUtils
from Utils import Utils
from copy import deepcopy
from Logger import Logger
import logging

class LinearRelaxation:
    __lp = None
    __io_utils = None


    def __init__(self, lp, io_utils):
        self.__lp = lp
        self.__io_utils = io_utils


    def solve_linear_relaxation(self, feasible_basis=[]):
        """
        Main method to solve a linear relaxation. All the possible situations are
        evaluated in order to decide which Simplex algorithm to use (Primal or Dual).

        All operations are stored in a log file.
        """

        linear_relaxation_output = {} # Dictionnary to store the results
        status         = ""
        message        = ""
        obj_value      = 0
        solution       = []
        certificate    = []
        #feasible_basis = []

        # Get the neg entries in both c and b arrays from the tableau
        index_neg_entry_in_c    = self.__lp.get_first_neg_entry_col_in_c()
        index_list_b_neg_values = self.__lp.get_b_neg_entries_rows()
        
        ############################################################################################
        # First we need to make some verifications so that Simplex will work fine without any loop #
        ############################################################################################

        # If there's a row with all variables = 0 and b != 0 (absurd). There is no solution then.
        is_inconsistent_row = self.__lp.is_any_row_inconsistent()

        if is_inconsistent_row > 0:
            message = "No solution possible. Equation " + str(is_inconsistent_row) + " is an absurd.\n\n"
            linear_relaxation_output["message"] = message
            
            # Saving logs
            print(message + self.__lp.print_tableau())
            self.__io_utils.write_output(message + self.__lp.print_tableau())
            logging.debug(message + self.__lp.print_tableau())
            exit(0)


        # The c array is in optimum state. Dual Simplex can be used
        if index_neg_entry_in_c < 0 and self.__lp.get_first_neg_entry_row_in_b() > 0:
            logging.debug(">> The c array is in optimum status: Dual Simplex will be used.")
            
            status, certificate, obj_value, feasible_basis, solution = self.__apply_dual_simplex(feasible_basis)
            message = str(status) + "\n"

            if status == Utils.LP_FEASIBLE_BOUNDED:
                message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)
                print(Logger.get_feasible_bounded_message(solution, obj_value, certificate)) # Log on the screen

            elif status == Utils.LP_INFEASIBLE:
                message = message + str(certificate)
                print(Logger.get_infeasible_message(certificate)) # Log on the screen
            else:
                message = ">>>> Something terrible happened. Dual Simplex should never lead to unbounded!"
                logging.debug(message)
                print(message)

            #self.__io_utils.write_output(message)
        else:
            logging.debug(">> The c array is not in optimum status or it is and there is no negative entry in b")
            logging.debug(">> Primal Simplex will be used.")

            #feasible_basis = []
            #if len(feasible_basis) == 0:
            #    feasible_basis.append(-1) # There's no pivot in the c row

            rows = self.__lp.get_tableau_num_rows()
            cols = self.__lp.get_tableau_num_cols()

            if not index_list_b_neg_values:
                logging.debug(">>>> All entries in b are non-negative. No need for an auxiliar LP.")
                # Store the initial columns that make the basic solution
                if len(feasible_basis) == 0:
                    feasible_basis.append(-1) # There's no pivot in the c row
                    for i in range(1, rows):
                       feasible_basis.append(cols - rows - 1 + i)

                # Apply Primal Simplex and get the result
                status,certificate,obj_value,feasible_basis,solution = self.__apply_primal_simplex(feasible_basis)

                # Store the string to print to the output file
                message = str(status) + "\n"
                if status == Utils.LP_FEASIBLE_BOUNDED:
                    message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)
                    print(Logger.get_feasible_bounded_message(solution, obj_value, certificate)) # Log on the screen
                elif status == Utils.LP_FEASIBLE_UNBOUNDED:
                    message = message + str(certificate)
                    print(Logger.get_feasible_unbounded_message(certificate)) # Log on the screen
                else:
                    message = ">>>> Something terrible happened. The objective value is negative, and that is such an heresy!"
                    loggin.debug(message)
                    print(message)

                # Store the final result in the output file
                #self.__io_utils.write_output(message)
            else:
                logging.debug(">>>> There are negative entries in b. An auxiliar LP is needed.")
                logging.debug(">>>> Creating auxiliar LP to find a feasible basic solution to the problem")

                feasible_basis = []
                feasible_basis.append(-1) # There's no pivot in the c row

                # Create an auxiliar LP to find a basic solution to the original problem
                simplex_aux = deepcopy(self)
                simplex_aux.__lp.make_auxiliar_lp(index_list_b_neg_values)
                
                # Pivotate the extra variables to prepare the auxiliar tableau for the primal simplex
                rows = simplex_aux.__lp.get_tableau_num_rows()
                cols = simplex_aux.__lp.get_tableau_num_cols()

                # Store the initial columns that make the basic solution
                for i in range(1, rows):
                    simplex_aux.__lp.pivotate_element(i, cols - rows + i - 1)
                    feasible_basis.append(cols - rows - 1 + i)

                # Primal Simplex for auxiliar LP is always feasible
                status,certificate,aux_obj_value,feasible_basis,sol = simplex_aux.__apply_primal_simplex(feasible_basis)

                if aux_obj_value == 0:
                    # Pivotate the base columns found in the auxiliar LP
                    for i in range(1, rows):
                        self.__lp.pivotate_element(i, feasible_basis[i])
                    
                    # Apply the Primal Simplex Algorithm and get the result
                    status,certificate,obj_value,feasible_basis,solution = self.__apply_primal_simplex(feasible_basis)
                    message = str(status) + "\n"
                    if status == Utils.LP_FEASIBLE_BOUNDED:
                        message = message + str(solution) + "\n" + str(obj_value) + "\n" + str(certificate)
                        # Log on the screen
                        print(Logger.get_feasible_bounded_message(solution, obj_value, certificate))
                    elif status == Utils.LP_FEASIBLE_UNBOUNDED:
                        message = message + str(certificate)
                        # Log on the screen
                        print(Logger.get_feasible_unbounded_message(certificate))

                    # Print the result to the output file
                    #self.__io_utils.write_output(message)
                    
                elif aux_obj_value < 0:
                    logging.debug(">>>> The original LP is infeasible.")
                    logging.debug(">>>> Infeasibility Certificate: " + str(certificate))

                    status  = Utils.LP_INFEASIBLE
                    message = str(Utils.LP_INFEASIBLE) + "\n" + str(certificate)

                    #self.__io_utils.write_output(message)
                    print(Logger.get_infeasible_message(certificate)) # Log on the screen
                else:
                    message = ">>>> Something terrible happened. The objective value is negative, and that is such an heresy!"
                    logging.debug(message)
                    print(message)

        linear_relaxation_output["message"]        = message
        linear_relaxation_output["status"]         = status
        linear_relaxation_output["certificate"]    = certificate
        linear_relaxation_output["obj_value"]      = obj_value
        linear_relaxation_output["solution"]       = solution
        linear_relaxation_output["feasible_basis"] = feasible_basis
        
        return linear_relaxation_output


    def __apply_primal_simplex(self, feasible_basis_columns):
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

                self.__lp.pivotate_element(row, col) # Pivotate the chosen element
                feasible_basis_columns[row] = col  # Update the base columns to the basic feasible solution
        
        # Check whether the LP is bounded
        if is_lp_bounded:
            #obj_value = round(self.__lp.get_objective_value(), 6)
            obj_value = self.__lp.get_objective_value()
            optimality_certificate = list(self.__lp.get_optimality_certificate())

            num_vars = self.__lp.get_tableau_num_cols() - 2*self.__lp.get_tableau_num_rows() + 1
            solution = [0]*num_vars # Starting solution with zeros
            for i in xrange(self.__lp.get_LP_init_column(), num_cols - num_rows):
                # If this column belongs to the base, then we add b solution to the variable its variable
                if i in feasible_basis_columns:
                    row_sol_base = feasible_basis_columns.index(i)
                    solution[i - num_rows + 1] = self.__lp.get_tableau_elem(row_sol_base, num_cols - 1)

            # Turn the values into float
            #for i in xrange(num_vars): 
            #    solution[i] = round(solution[i], 6)
            for i in xrange(len(optimality_certificate)):
                optimality_certificate[i] = round(optimality_certificate[i], 6)

            logging.debug(IOUtils.print_header_line_screen())
            logging.debug(">> Maximum objective value: " + str(obj_value))
            logging.debug(">> Solution: "                + str(solution))
            logging.debug(">> Optimality certificate: "  + str(optimality_certificate))
            logging.debug(IOUtils.print_header_line_screen())

            # Return the id for a feasible bounded solution and the base columns associated
            return (Utils.LP_FEASIBLE_BOUNDED, optimality_certificate, obj_value, feasible_basis_columns, solution)
        else:
            num_vars = num_cols - num_rows
            unbounded_certificate = [0]*num_vars
            unbounded_certificate[col - num_rows + 1] = 1

            for i in xrange(self.__lp.get_LP_init_column(), num_cols - 1):
                if i in feasible_basis_columns:
                    cert_row = feasible_basis_columns.index(i)
                    neg_column_elem = -self.__lp.get_tableau_elem(cert_row, col)
                    unbounded_certificate[i - num_rows + 1] = float(neg_column_elem)
            
            for i in xrange(num_vars): # Turn the values into float
                unbounded_certificate[i] = round(unbounded_certificate[i], 6)
            
            logging.debug(">>>> Unbounded certificate: " + str(unbounded_certificate))
            return (Utils.LP_FEASIBLE_UNBOUNDED, unbounded_certificate, 0, None, None)


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


    def __apply_dual_simplex(self, feasible_basis=[]):
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

        # If we do not have an initial feasible basis
        if len(feasible_basis) == 0:
            feasible_basis = [-1]*num_rows

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
                break
            else:
                logging.debug(">>>>>> The element chosen is " + str(self.__lp.get_tableau_elem(row, col))
                                     + " at the position (" + str(row) + ", " + str(col) + ")")

                self.__lp.pivotate_element(row, col) # Pivotate the chosen element
                feasible_basis[row] = col

        # Check whether the LP is feasible
        if is_lp_feasible:
            obj_value = round(self.__lp.get_objective_value(), 6)
            optimality_certificate = list(self.__lp.get_optimality_certificate())

            num_vars = num_cols - 2*num_rows + 1
            solution = [0]*num_vars # Starting solution with zeros
            for i in xrange(self.__lp.get_LP_init_column(), num_cols - num_rows):
                # If this column belongs to the base, then we add b solution to the variable its variable
                if i in feasible_basis:
                    row_sol_base = feasible_basis.index(i)
                    solution[i - num_rows + 1] = self.__lp.get_tableau_elem(row_sol_base, num_cols - 1)

            # Turn the values into float
            #for i in xrange(num_vars): 
            #    solution[i] = round(solution[i], 6)
            for i in xrange(len(optimality_certificate)):
                optimality_certificate[i] = round(optimality_certificate[i], 6)

            logging.debug(IOUtils.print_header_line_screen())
            logging.debug(">> Maximum objective value: " + str(obj_value))
            logging.debug(">> Solution: "                + str(solution))
            logging.debug(">> Optimality certificate: "  + str(optimality_certificate))
            logging.debug(IOUtils.print_header_line_screen())

            return (Utils.LP_FEASIBLE_BOUNDED, optimality_certificate, obj_value, feasible_basis, solution)
        else:
            # Get the infeasibility certificate from the operation matrix
            infeasible_certificate = self.__lp.get_tableau()[row, 0:num_rows - 1]
            # Turn the values into float
            for i in xrange(len(infeasible_certificate)):
                infeasible_certificate[i] = round(infeasible_certificate[i], 6)

            return (Utils.LP_INFEASIBLE, infeasible_certificate, 0, None, None)


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