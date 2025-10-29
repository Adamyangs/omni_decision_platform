import numpy as np
from scipy.optimize import linprog

# Define the coefficients of the objective function
c = [1]

# Define the coefficients of the inequality constraints
A_ub = [[1, 1]]
b_ub = [300]

# Define the bounds for the variables
bounds = [(0, None), (0, None)]

# Solve the linear programming problem
res = linprog(c, A_ub, b_ub, bounds)

print("The maximum value of A is: ", res.fun)