from lcp import LinearComplementarityProblem
import numpy as np

# Example 1
M2 = np.array([[2,1], [-1,2]])
q2 = np.array([2, 1])
lcp = LinearComplementarityProblem(M2, -q2)
lcp.solve_with_lemke_method(max_itr=10, verbose=True)

# Example 2
# The example 2.8 of Murty's book. Should lead to a feasible solution
M4 = -np.array([[-1,1,1,1], [1,-1,1,1], [-1,-1,-2,0], [-1,-1,0,-2]])
q4 = np.array([3, 5, -9, -5])
lcp = LinearComplementarityProblem(M4, q4)
lcp.solve_with_lemke_method(max_itr=10, verbose=True)

# Example 3
# The example 2.9 of Murty's book. Should lead to ray termination
M3 = -np.array([[1,0,3], [-1,2,5], [2,1,2]])
q3 = np.array([-3, -2, -1])
lcp = LinearComplementarityProblem(M3, q3)
lcp.solve_with_lemke_method(max_itr=10, verbose=True)