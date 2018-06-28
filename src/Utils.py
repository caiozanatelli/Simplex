from copy import deepcopy

class Utils:
    LP_INFEASIBLE         = 0
    LP_FEASIBLE_UNBOUNDED = 1
    LP_FEASIBLE_BOUNDED   = 2

    @classmethod
    def get_header_line_screen(cls):
        """
        Static method used to draw a line.
        """
        return "################################################################################"


    @classmethod
    def get_float_formatted_array(cls, array):
        formatted_array = deepcopy(array)
        for i in xrange(len(formatted_array)):
            formatted_array[i] = round(formatted_array[i], 6)

        return formatted_array