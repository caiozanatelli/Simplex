import numpy as np
from fractions import Fraction
from LinearProgramming import LinearProgramming
from LinearRelaxation import LinearRelaxation
from IOUtils import IOUtils
from Utils import Utils
from copy import deepcopy
from Logger import Logger
import logging
import math

class IntegerProgramming:
    __lp = None
    __io_utils = None
    __max_obj_value = None
    __max_solution = None

    def __init__(self, lp, io_utils):
        self.__lp = lp
        self.__io_utils = io_utils



    def solve_cutting_planes(self, feasible_basis):
        list_equations_used = []
        integer_programming_output = {}

        while True:
            floor_row = self.__lp.get_first_row_frac_in_b(feasible_basis, list_equations_used)

            # There is no frac row in b, so the solution is already integer
            if floor_row == -1:
                break
            else:
                print(Logger.get_adding_new_cutting_plane_message())
                logging.debug(Logger.get_adding_new_cutting_plane_message())

                new_row = self.__lp.get_floor_row(floor_row)

                self.__lp.add_restriction(new_row)
                self.__uptadate_feasible_basis_after_adding_restriction(self.__lp, feasible_basis)

                for row in xrange(1, len(feasible_basis)):
                    col = feasible_basis[row]
                    self.__lp.pivotate_element(row, col)

                linear_relaxation = LinearRelaxation(self.__lp, self.__io_utils)
                linear_relaxation_output = linear_relaxation.solve_linear_relaxation(feasible_basis)

                solution = linear_relaxation_output["solution"]
                solution = self.__lp.get_solution_from_feasible_basis(solution, feasible_basis)

                integer_programming_output["obj_value"] =  linear_relaxation_output["obj_value"]
                integer_programming_output["solution"]  =  solution


        # Return the integer programming output
        return integer_programming_output


    def solve_branch_and_bound(self, feasible_basis, solution, obj_value):
        """
        Main method to solve an integer programming through the Branch and Bound Algorithm.
        """
        branch_and_bound_solution = self.__solve_recursive_branch_and_bound(self.__lp, feasible_basis, solution, obj_value, 0)
        return branch_and_bound_solution


    def __solve_recursive_branch_and_bound(self, tableau, feasible_basis, curr_solution, curr_obj_value, max_obj_value, 
                                            status=None, max_solution=None, stop_recursion=False):
        """
        A recursive implementation for the Branch and Bound algorithm. It returns the status (Feasible or not), the maximum
        integer objective value and its associated solution.
        """
        output = {}
        output["status"]         = 0
        output["solution"]       = []
        output["obj_value"]      = 0
        output["feasible_basis"] = []
        output["stop_recursion"] = True

        if stop_recursion == False:
            tableau_frac_index_sol, b_value = self.__is_solution_integer(tableau, curr_solution)

            if tableau_frac_index_sol == -1: # The solution and the obj. value are both integer

                if self.__max_obj_value == None or self.__max_obj_value < curr_obj_value:
                    self.__max_obj_value  = curr_obj_value
                    self.__max_solution   = curr_solution
                
                #output = {}
                #output["status"]         = Utils.LP_FEASIBLE_BOUNDED
                #output["solution"]       = max_solution
                #output["obj_value"]      = max_obj_value
                #output["feasible_basis"] = feasible_basis
                #output["stop_recursion"] = True

                #return output

            else:
                # Left tableau: floor(x)  restriction
                new_tableau_left  = deepcopy(tableau)
                new_tableau_right = deepcopy(tableau)

                floor_value = Fraction(math.floor(b_value))
                ceil_value  = Fraction(math.ceil(b_value))

                restriction_row_left  = self.__get_new_restriction_branch_and_bound(new_tableau_left, tableau_frac_index_sol, floor_value)
                restriction_row_right = self.__get_new_restriction_branch_and_bound(new_tableau_right, tableau_frac_index_sol, ceil_value)
                
                new_tableau_left.add_restriction(restriction_row_left, 1)
                new_tableau_right.add_restriction(restriction_row_right, -1)

                new_left_basis  = deepcopy(feasible_basis)
                new_right_basis = deepcopy(feasible_basis)

                self.__uptadate_feasible_basis_after_adding_restriction(new_tableau_left, new_left_basis)
                self.__uptadate_feasible_basis_after_adding_restriction(new_tableau_right, new_right_basis)

                for row in xrange(1, len(new_left_basis)):
                    col = new_left_basis[row]
                    new_tableau_left.pivotate_element(row, col)

                for row in xrange(1, len(new_right_basis)):
                    col = new_right_basis[row]
                    new_tableau_right.pivotate_element(row, col)

                # Solve linear relaxation for the left branch (<=)
                simplex_left = LinearRelaxation(new_tableau_left, self.__io_utils)
                simplex_left_output = simplex_left.solve_linear_relaxation(new_left_basis)

                simplex_left_status = simplex_left_output["status"]
                new_left_solution   = simplex_left_output["solution"]
                new_left_obj_value  = simplex_left_output["obj_value"]
                simplex_left_stop_recursion = False

                if simplex_left_status == Utils.LP_INFEASIBLE:
                    simplex_left_stop_recursion  = True
                elif simplex_left_status == Utils.LP_FEASIBLE_BOUNDED:
                    if new_left_obj_value <= self.__max_obj_value:
                        simplex_left_stop_recursion  = True
                    else:
                        print(Logger.get_new_branch_message("LEFT"))
                        logging.debug(Logger.get_new_branch_message("LEFT"))

                        self.__solve_recursive_branch_and_bound(new_tableau_left, new_left_basis, new_left_solution, new_left_obj_value, 
                                                   max_solution, simplex_left_status, max_obj_value, simplex_left_stop_recursion)

                # Solve the linear relaxation for the right branch (>=)
                simplex_right = LinearRelaxation(new_tableau_right, self.__io_utils)
                simplex_right_output = simplex_right.solve_linear_relaxation(new_right_basis)
                simplex_right_status = simplex_right_output["status"]
                new_right_solution   = simplex_right_output["solution"]
                new_right_obj_value  = simplex_right_output["obj_value"]
                simplex_right_stop_recursion = False

                if simplex_right_status == Utils.LP_INFEASIBLE:
                    simplex_right_stop_recursion  = True
                elif simplex_right_status == Utils.LP_FEASIBLE_BOUNDED:

                    if new_right_obj_value <= self.__max_obj_value:
                        simplex_right_stop_recursion  = True
                    else:
                        print(Logger.get_new_branch_message("RIGHT"))
                        logging.debug(Logger.get_new_branch_message("RIGHT"))

                        self.__solve_recursive_branch_and_bound(new_tableau_right, new_right_basis, new_right_solution, new_right_obj_value, 
                                                             max_obj_value, simplex_right_status, max_solution, simplex_right_stop_recursion)


        #else:
        output["status"]         = Utils.LP_FEASIBLE_BOUNDED
        output["solution"]       = self.__max_solution
        output["feasible_basis"] = feasible_basis
        output["obj_value"]      = self.__max_obj_value
        output["stop_recursion"] = True

        return output


    def __is_solution_integer(self, tableau, solution):
        """
        Check whether a solution is integer or not. If so, return the column in the tableau
        where the first fraction solution appears, and -1 otherwise.
        """
        for i in xrange(len(solution)):
            if solution[i].denominator != 1:
                return i + tableau.get_LP_init_column(), solution[i]
        return -1, None

    
    def __uptadate_feasible_basis_after_adding_restriction(self, tableau, feasible_basis):
        """
        Update the feasible column basis after adding a restriction. This is necessary because
        we add a column in the operation matrix during the process, and hence we must add 1 to
        the column basis indexes.
        """
        for row in xrange(1, len(feasible_basis)):
            feasible_basis[row] = feasible_basis[row] + 1

        feasible_basis.append(tableau.get_tableau_num_cols() - 2)



    def __get_new_restriction_branch_and_bound(self, tableau, var_col, b_value):
        """
        Build a new restriction for the branch and bound algorithm.
        """
        restriction_row = np.zeros((tableau.get_tableau_num_cols())).astype('object')
        restriction_row[var_col] = 1
        restriction_row[tableau.get_tableau_num_cols() - 1] = b_value

        for i in xrange(0, len(restriction_row)):
            restriction_row[i] = Fraction(restriction_row[i])

        return restriction_row 

