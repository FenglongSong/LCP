import numpy as np
import casadi as ca

def is_lexico_positive(matrix: np.ndarray, zero_tol=1e-10) -> bool:
      """ Tell if a matrix or vector is lexico positive.
      
      A vector is lexico positive if its first non-zero elements is strictly positive,
      a matrix is lexico positive if all its rows are lexico positive.

      Args:
            matrix: a ndarray of size [m,n]. The array can be a 1d array representing a vector.

      Returns:
            a bool variable implying the matrix is lexico positive or not.
      """

      # process the dimension of inputs
      if matrix.size == 0:
            raise ValueError("The input matrix is empty!")
      if len(matrix.shape) > 2:
            raise ValueError("Wrong array size!")
      if len(matrix.shape) == 1:
            matrix = np.reshape(matrix, (1, matrix.size))

      [m, n] = matrix.shape
      undetermined_row_indices = list(range(m))
      tmp_matrix = matrix
      for j in range(n):
            tmp_matrix = tmp_matrix[undetermined_row_indices, :]
            col = tmp_matrix[:, j]
            if (col > 0).all():
                  return True       # if all elements in this column is positive
            elif (col < 0).any():
                  return False      # if at least one element in this column is negative
            else:
                  undetermined_row_indices = [i for i ,e in enumerate(col) if e == 0]     # ! maybe e==0 is not a good way

      return False


def lexico_argmin(matrix: np.array, zero_tol=1e-10) -> int | list[int]:
      """ Find the index of lexico minimum of a matrix.
      
      The lexico minimum of a set of vectors {a[1], a[2],..., a[m]} is the vector a[j] satisfying
      that (a[i] - a[j]) is lexico positive for all i in {1,2,...,m}/{j}.

      Args:
            matrix: a ndarray of size [m,n].

      Returns:
            the index of lexico minimum row.
      """
      # TODO: can be improved for better efficiency, refer to https://stackoverflow.com/questions/13544476/how-to-find-max-and-min-in-array-using-minimum-comparisons
      if matrix.size == 0:
            raise ValueError("The input matrix is empty!")
      if len(matrix.shape) > 2:
            raise ValueError("Wrong array size!")
      if len(matrix.shape) == 1:
            matrix = np.reshape(matrix, (1, matrix.size))

      [m, n] = matrix.shape
      argmin = 0
      min = matrix[0, :]
      for i in range(m):
            a = matrix[i, :]
            if is_lexico_positive(min - a):
                  argmin = i
                  min = a

      return argmin

class HyperplaneRepresentation:
      """
      Define a H-reprenestation of a polyhedron.
      """
      
      def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
            if A.shape[0] != b.shape[0]:
                  raise ValueError("HRepresentation: Dimension inconsistency between A and b!")

            self.A = A
            self.b = b
            self.m = A.shape[0] # number of affine constraints
            self.n = A.shape[1] # number of dimension

      def remove_redundant_constraints(self) -> None:
            """ Remove all the redundant constraints.
            
            Reference:
                  Section 2.20 of https://people.inf.ethz.ch/fukudak/Doc_pub/polyfaq220115c.pdf
            
            We want to test whether the subsystem of first m-1 inequalities Cx \leq d implies the last inequality s x \leq t. 
            If so, the inequality s x \leq t is redundant and can be removed from the system. A linear programming (LP) 
            formulation of this checking is rather straightforward:

            f^* = max s^T x
                  s.t C x \leq d,
                      s x \leq t + 1
            Then the inequality s x \leq t is redundant if and only if the optimal value f^* is less than or equal to t.
            """
            
            i = 0
            while True:
                  s = self.A[[i], :]
                  t = self.b[i]

                  C = np.vstack((self.A[:i, :], self.A[i+1:, :]))
                  d = np.concatenate((self.b[:i], self.b[i+1:]))

                  opti = ca.Opti()
                  x = opti.variable(self.n)
                  opti.subject_to(C @ x <= d)
                  opti.subject_to(s @ x <= t+1)
                  opti.minimize(-s@x)

                  opts = {'ipopt.print_level':0, 'print_time':0}
                  opti.solver('ipopt', opts)

                  sol = opti.solve()
                  if not sol.stats()['success']:
                        raise RuntimeError("Ipopt failure.")
                  
                  x_opt = sol.value(x)
                  f_opt = s @ x_opt
                  if f_opt <= t: # redundant
                        print("Found a redundant constraint ", s)
                        self.A = C
                        self.b = d
                        self.m -= 1
                  else:
                        i += 1

                  if i == self.m:
                        break

                  # A very naive way to avoid dead cycle
                  if i >= 5000:
                        raise RuntimeError("Dead cycle.")