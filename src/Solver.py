from IOUtils import IOUtils
from LinearProgramming import LinearProgramming
from Simplex import Simplex

LP_DIR_IN   = "../tests/toys/teste3.txt"
RES_DIR_OUT = "../tests/toys/saida_teste3.txt"

if __name__ == '__main__':
    io = IOUtils(LP_DIR_IN, RES_DIR_OUT)
    #input_matrix = io.read_input("../tests/spec_test.txt")
    #input_matrix = io.read_input("../tests/aux_pl.txt")
    #input_matrix = io.read_input("../tests/aux_pl_marzano.txt")

    rows, cols, input_matrix = io.read_input()

    simplex = Simplex(rows, cols, input_matrix, io)
    simplex.solve()
