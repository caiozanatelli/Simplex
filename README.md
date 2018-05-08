# Simplex
An implementation of the Simplex algorithm for solving linear optimization problems. This approach covers both Primal and Dual Simplex.

# How to Run

To run this program, you should call it like this:

```bash

$ ./run.sh entrada.txt

```

You can also call the Python program directly from the src directory. For that, use this command:

```bash

$ python src/Solver.py entrada.txt

```

# Output

The output consists of two files that are created in the same directory where the program was called:

* **conclusao.txt**: stores the final result of the linear programming problem, such as certificates, solutions, and objective value, when applicable.

* **log_simplex.txt**: stores all the intermediate phases made during the Simplex algorithm, such as pivotation and all the operations on the tableau.
