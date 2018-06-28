import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from LinearRelaxation import LinearRelaxation
from IntegerProgramming import IntegerProgramming
from Utils import Utils
from IOUtils import IOUtils
from copy import deepcopy
from Logger import Logger
import logging

class Simplex:
    SOLVER_MODE_INT_CUTTING_PLANES   = 0
    SOLVER_MODE_INT_BRANCH_AND_BOUND = 1
    SOLVER_MODE_LINEAR_PROGRAMMING   = 3

    __lp                     = None
    __obj_func_c             = None
    __feasible_base_solution = None
    __io_utils               = None


    def __init__(self, num_rows, num_cols, input_matrix, io_utils):
        logging.debug(IOUtils.print_header_line_screen())
        logging.debug(">> Starting Simplex.")
        logging.debug(IOUtils.print_header_line_screen())

        self.__io_utils = io_utils
        # Create the tableau that represents the linear programming
        self.__lp = LinearProgramming(num_rows, num_cols, input_matrix)
        # Set the c objective function and the b array
        self.__obj_func_c = input_matrix[0][:-1]



    def solve(self, solver_mode):
        """
        Main method to solve a linear programming. The solver mode param indicates
        which type of linear programming solver  we want to use: linear relaxation
        or integer programming. The first one obviously concerns the real solution
        for the linear programming, whilst  the other restricts the solution to be
        integer.
        """
        print(Logger.get_solve_algorithm_message("Linear Relaxation"))
        logging.debug(Logger.get_solve_algorithm_message("Linear Relaxation"))

        linear_relaxation = LinearRelaxation(self.__lp, self.__io_utils)
        linear_relaxation_output = linear_relaxation.solve_linear_relaxation()
        
        message_lin_rlx        = linear_relaxation_output["message"]
        status_lin_rlx         = linear_relaxation_output["status"]
        certificate_lin_rlx    = linear_relaxation_output["certificate"]
        obj_value_lin_rlx      = linear_relaxation_output["obj_value"]
        solution_lin_rlx       = linear_relaxation_output["solution"]
        feasible_basis_lin_rlx = linear_relaxation_output["feasible_basis"]

        #final_message_output   = str(status_lin_rlx) + "\n"
        final_output_message   = str(status_lin_rlx)

        if status_lin_rlx == Utils.LP_INFEASIBLE:             # The LP is infeasible
            pass
        elif status_lin_rlx == Utils.LP_FEASIBLE_UNBOUNDED:   # The LP is feasible but unbounded
            final_output_message = final_output_message + "\n" + str(certificate_lin_rlx)
        elif status_lin_rlx == Utils.LP_FEASIBLE_BOUNDED:     # The LP is feasible and bounded
            
            integer_programming   = IntegerProgramming(self.__lp, self.__io_utils)
            integer_solution  = []
            integer_obj_value = 0
            
            if solver_mode == Simplex.SOLVER_MODE_INT_CUTTING_PLANES:
                # Saving logs
                print(Logger.get_solve_algorithm_message("Cutting Planes"))
                logging.debug(Logger.get_solve_algorithm_message("Cutting Planes"))

                # Solving the lp through the cutting planes algorithm
                cutting_planes_output = integer_programming.solve_cutting_planes(feasible_basis_lin_rlx)
                integer_solution  = cutting_planes_output["solution"]
                integer_obj_value = cutting_planes_output["obj_value"]

                final_output_message = Logger.get_integer_programming_solution_message(status_lin_rlx, 
                                                obj_value_lin_rlx, solution_lin_rlx , certificate_lin_rlx, 
                                                integer_obj_value, integer_solution)


            elif solver_mode == Simplex.SOLVER_MODE_INT_BRANCH_AND_BOUND:
                # Saving logs
                print(Logger.get_solve_algorithm_message("Branch and Bound"))
                logging.debug(Logger.get_solve_algorithm_message("Branch and Bound"))

                # Solving the lp through the branch and bound algorithm
                branch_and_bound_output = integer_programming.solve_branch_and_bound(feasible_basis_lin_rlx, 
                                                                        solution_lin_rlx, obj_value_lin_rlx)
                integer_solution  = branch_and_bound_output["solution"]
                integer_obj_value = branch_and_bound_output["obj_value"] 

                final_output_message = Logger.get_integer_programming_solution_message(status_lin_rlx, 
                                           obj_value_lin_rlx, solution_lin_rlx, certificate_lin_rlx, 
                                           integer_obj_value, integer_solution)

            # Print the conclusion log on the screen
            log_message = "\n" + Logger.get_conclusion_header_message() + Logger.get_feasible_bounded_message(integer_solution, integer_obj_value, certificate_lin_rlx)
            print(log_message)
            logging.debug(log_message)

        # Save the final output in the conclusion file
        self.__io_utils.write_output(final_output_message)
