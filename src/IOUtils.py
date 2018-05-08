import numpy as np
from fractions import Fraction

class IOUtils:
    """
    Class for input/output manipulation.
    """

    __file_in  = None
    __file_out = None

    def __init__(self, file_dir_in, file_dir_out):
        self.__file_in  = open(file_dir_in, 'r')
        self.__file_out = open(file_dir_out, 'w')
    

    def __del__(self):
        self.__file_in.close()
        self.__file_out.close()


    def read_input(self):
        """
        Read the input according to the specification.
        """
        n = int(self.__file_in.readline())
        m = int(self.__file_in.readline())

        matrix_str = self.__file_in.read()
        matrix = np.array(eval(matrix_str)).astype('object')

        return n, m, matrix


    def write_output(self, message):
        """
        Write a message to the output file.
        """
        self.__file_out.write(message)


    @classmethod
    def print_header_line_screen(cls):
        """
        Static method used to draw a line.
        """
        return "################################################################################"