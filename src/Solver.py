from IOUtils import IOUtils
from LinearProgramming import LinearProgramming
from Simplex import Simplex

if __name__ == '__main__':
    io = IOUtils()
    input_matrix = io.read_input("../tests/spec_test.txt")

    #lp = LinearProgramming(2, 3, input_matrix)

    simplex = Simplex(2, 3, input_matrix)
    simplex.solve()
    
    #lp.__make_tableau(input_matrix)
    #lp.make_tableau(input_matrix)
