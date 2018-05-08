# Simplex
An implementation for the Simplex algorithm for solving linear optimization problems. This approach covers both Primal and Dual Simplex.

# How to Run

To run this program, you should call the program like this:

```bash

$ ./run.sh entrada.txt

```

Or you could call the Python program directly from the src directory. For that, use this command:

```bash

$ python src/Solver.py entrada.txt

```

# Output

The output is consisted of two files. The first one is named "conclusao.txt", and is responsable for storing the final result for the linear programming. Moreover, the file entitled "log_simplex" stores all the operations made during the Simplex process. This files are both stored in the same directory where the program was called.
