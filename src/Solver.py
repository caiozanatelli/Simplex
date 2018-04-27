from IOUtils import IOUtils
from LinearProgramming import LinearProgramming

if __name__ == '__main__':
    io = IOUtils()
    input_matrix = io.read_input("../tests/spec_test.txt")

    lp = LinearProgramming(3, 4)
    lp.make_tableau(input_matrix)