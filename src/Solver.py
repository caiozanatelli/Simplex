from IOUtils import IOUtils
from LinearProgramming import LinearProgramming
from Simplex import Simplex

if __name__ == '__main__':
    io = IOUtils()
    #input_matrix = io.read_input("../tests/spec_test.txt")
    input_matrix = io.read_input("../tests/aux_pl.txt")
    simplex = Simplex(2, 3, input_matrix)
    simplex.solve()
