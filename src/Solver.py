from IOUtils import IOUtils
from LinearProgramming import LinearProgramming
from Simplex import Simplex

LP_DIR = "../tests/aux_pl_marzano.txt"

if __name__ == '__main__':
    io = IOUtils()
    #input_matrix = io.read_input("../tests/spec_test.txt")
    #input_matrix = io.read_input("../tests/aux_pl.txt")
    #input_matrix = io.read_input("../tests/aux_pl_marzano.txt")

    rows, cols, input_matrix = io.read_input(LP_DIR)

    simplex = Simplex(rows, cols, input_matrix)
    simplex.solve()
