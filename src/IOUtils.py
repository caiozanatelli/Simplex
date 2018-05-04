import numpy as np
from fractions import Fraction

class IOUtils:
    
    @classmethod
    def print_header_line_screen(cls):
        print("###################################################################")

    def read_input(self, file_dir):
        fp = open(file_dir, 'r')
        
        n = int(fp.readline())
        m = int(fp.readline())

        print("Number of lines:   " + str(n)
              + "\nNumber of columns: " + str(m))

        matrix_str = fp.read()
        matrix = np.array(eval(matrix_str)).astype('object')

        #for i in range(matrix.shape[0]):
        #    for j in range(matrix.shape[1]):
        #        matrix[i][j] = Fraction(matrix[i][j])

        return matrix
