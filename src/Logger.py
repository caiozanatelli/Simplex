import logging
import numpy as np
from fractions import Fraction
from copy import deepcopy
from Utils import Utils

class Logger:


    @classmethod
    def draw_header_line_screen(cls):
        """
        Static method used to draw a line.
        """
        return "################################################################################"


    @classmethod
    def get_feasible_bounded_message(cls, solution, obj_value, certificate):
        message = "##############################\n"
        message = message + ". : Feasible and Bounded : .\n"
        message = message + "##############################\n"
        message = message + "[+] Solution: " + str(Utils.get_float_formatted_array(solution)) + "\n"  
        message = message + "[+] Objective Value: " + str(obj_value) + "\n" 
        message = message + "[+] Optimality Certificate: " + str(certificate) + "\n"
        return message


    @classmethod
    def get_feasible_unbounded_message(cls, certificate):
        message = "################################\n"
        message = message + " . : Feasible and Unbounded : .  \n"
        message = message + "################################ \n"
        message = message + "[+] Unbounded Certificate: " + str(certificate) + "\n"
        return message


    @classmethod
    def get_infeasible_message(cls, certificate):
        message = "####################\n"
        message = message + " . : Infeasible : .\n"
        message = message + "####################\n"
        message = message + "[+] Infeasibility Certificate: " + str(certificate) + "\n"
        return message


    @classmethod
    def get_integer_programming_solution_message(cls, status, linear_rlx_obj_value, linear_rlx_solution,
                                                 linear_rlx_opt_cert, int_prog_obj_value, int_prog_solution):
        message = str(status) + "\n"
        message = message + str(Utils.get_float_formatted_array(int_prog_solution)) + "\n"
        message = message + str(int_prog_obj_value)   + "\n"
        message = message + str(Utils.get_float_formatted_array(linear_rlx_solution))  + "\n"
        message = message + str(round(linear_rlx_obj_value, 6)) + "\n"
        message = message + str(linear_rlx_opt_cert)

        return message


    @classmethod
    def get_conclusion_header_message(cls):
        return """\n%s\n \
                    . : Conclusion : . \n%s\n\n""" \
                    % (Logger.draw_header_line_screen(), Logger.draw_header_line_screen())


    @classmethod
    def get_solve_algorithm_message(cls, algorithm):
        return """\n%s\n \
                    . : Solving %s : . \n%s\n\n""" \
                    % (Logger.draw_header_line_screen(), algorithm, Logger.draw_header_line_screen())


    @classmethod
    def get_adding_new_cutting_plane_message(cls):
        return Utils.get_header_line_screen() + "\n Adding a new cutting plane \n"


    @classmethod
    def get_new_branch_message(cls, direction):
        return Utils.get_header_line_screen() + "\n Starting new %s branch \n" % direction