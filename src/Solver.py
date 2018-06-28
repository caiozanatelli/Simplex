from IOUtils import IOUtils
from LinearProgramming import LinearProgramming
from Simplex import Simplex
import logging
import sys

#LP_DIR_IN   = "../tests/toys/teste6.txt"
RES_DIR_OUT = "conclusao.txt"
LOG_DIR     = "log_simplex.txt"

if __name__ == '__main__':
    # Setting logger to register all the operations made in the Simplex Algorithm
    logging.basicConfig(filename = LOG_DIR, level = logging.DEBUG, format='%(message)s', filemode='w')
    logging.getLogger()

    if len(sys.argv) < 2:
        print("No input has been set. Aborting program.")
        exit(0)

    # Get the input file through a parameter
    input_file = sys.argv[1]

    # Reading the input
    io = IOUtils(input_file, RES_DIR_OUT)
    alg_mode, rows, cols, input_matrix = io.read_input()

    # Solving the linear programming through Simplex Algorithm
    simplex = Simplex(rows, cols, input_matrix, io)
    simplex.solve(alg_mode)
