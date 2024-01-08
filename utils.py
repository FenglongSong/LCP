import numpy as np
import casadi as ca

def is_complementary_basis(indices: list, n: int) -> bool:
	""" 
	Tell if the current basis is a complementary basis.
	"""
	for i in range(n):
		if indices[i] + n in indices:
			return False
		elif indices[i] - n in indices:
			return False
	return True

# TODO: implement and test it
def is_almost_complementary_basis(indices: list, n: int) -> bool:
	""" 
	Tell if the current basis is an almost complementary basis.
	"""
	return NotImplementedError


class Basis:
	"""
    Define the basis of a Linear Complementarity Problem (LCP).
    """
	def __init__(self, indices: list[int], n: int) -> None:
		# check if there are repeated elements in the indices list
		if len(set(indices)) < len(indices):
			raise ValueError("There are repeated elements in indices list.")
		
		# check if the elements of indices are within [0, 2n-1]
		if np.any(np.array(indices) < 0) or np.any(np.array(indices) > 2*n-1):
			raise ValueError("Invalid value in indices list.")
		
		self.n = n
		self.indices = indices
		
	
	def is_complementary_basis(self) -> bool:
		""" 
		Tell if the current basis is a complementary basis.
        """
		for i in range(self.n):
			if self.indices[i] + self.n in self.indices:
				return False
			elif self.indices[i] - self.n in self.indices:
				return False
		return True
	
	# TODO: implement and test it
	def is_almost_complementary_basis(self) -> bool:
		return NotImplementedError
	
	

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
