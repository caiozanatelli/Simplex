import numpy as np
from fractions import Fraction

class IOUtils:

    __file_in  = None
    __file_out = None

    def __init__(self, file_dir_in, file_dir_out):
        self.__file_in  = open(file_dir_in, 'r')
        self.__file_out = open(file_dir_out, 'w')
    

    def __del__(self):
        self.__file_in.close()
        self.__file_out.close()


    def read_input(self):
        n = int(self.__file_in.readline())
        m = int(self.__file_in.readline())

        print("Number of lines:   " + str(n)
              + "\nNumber of columns: " + str(m))

        matrix_str = self.__file_in.read()
        matrix = np.array(eval(matrix_str)).astype('object')

        return n, m, matrix


    def write_output(self, message):
        self.__file_out.write(message)


    @classmethod
    def print_header_line_screen(cls):
        print("###################################################################")