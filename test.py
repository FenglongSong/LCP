from lcp import LinearComplementarityProblem
from utils import *
import numpy as np
import unittest

class TestLexicoPositive(unittest.TestCase):

	def test_lexico_positivity(self):
		vec1 = np.array([0., 3., 4.])
		vec2 = np.array([-1., 0., 3., 5.])
		vec3 = np.array([0., 0., 0., 0.])
		vec4 = np.array([0., 0., 0., 0., 1e-6])
		vec5 = np.array([0., 1e-6, -1e-6, 1e3, -1e5])

		self.assertTrue(is_lexico_positive(vec1))
		self.assertFalse(is_lexico_positive(vec2))
		self.assertFalse(is_lexico_positive(vec3))
		self.assertTrue(is_lexico_positive(vec4))
		self.assertTrue(is_lexico_positive(vec5))

	def test_lexico_minimum(self):
		mat1 = np.array([[0, 1, -1, 0], [0, 0, 0, 1], [-3, 0, 0, 0]])
		mat2 = np.array([[1, 1, -1, 0], [0, 0, 0, 1], [3, 0, 0, 0]])
		mat3 = np.array([[-1, 1, -1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
		self.assertEqual(lexico_argmin(mat1), 2)
		self.assertEqual(lexico_argmin(mat2), 1)
		self.assertEqual(lexico_argmin(mat3), 0)


class TestLemkeMethod(unittest.TestCase):

	def test_feasible_example(self):
		# A simple example. Should lead to feasible solution, w=q, z=0
		M1 = np.array([[2,1], [-1,2]])
		q1 = np.array([2, 1])
		lcp = LinearComplementarityProblem(M1, q1)
		sol, status = lcp.solve_with_lemke_method(max_itr=10, verbose=False)
		self.assertTrue(np.linalg.norm(sol - np.concatenate((q1, np.zeros(2)))) < 1e-8)
		self.assertEqual(status, 0)

		# The example 2.8 of Murty's book. Should lead to a feasible solution w=0, z=[2,1,3,1]
		M2 = -np.array([[-1,1,1,1], [1,-1,1,1], [-1,-1,-2,0], [-1,-1,0,-2]])
		q2 = np.array([3, 5, -9, -5])
		lcp = LinearComplementarityProblem(M2, q2)
		sol, status = lcp.solve_with_lemke_method(max_itr=10, verbose=False)
		sol_true = np.array([0,0,0,0,2,1,3,1])
		self.assertTrue(np.linalg.norm(sol - sol_true) < 1e-8)
		self.assertEqual(status, 0)

	def test_ray_termination_example(self):
		# The example 2.9 of Murty's book. Should lead to ray termination
		M3 = -np.array([[1,0,3], [-1,2,5], [2,1,2]])
		q3 = np.array([-3, -2, -1])
		lcp = LinearComplementarityProblem(M3, q3)
		sol, status = lcp.solve_with_lemke_method(max_itr=10, verbose=False)
		self.assertEqual(status, 1)

	def test_degenerate_example(self):
		# The example in Section 2.2.8 of Murty's book. Should lead to cycling without using lexico perturbation
		M4 = np.array([[1,2,0], [0,1,2], [2,0,1]])
		q4 = np.array([-1, -1, -1])
		lcp = LinearComplementarityProblem(M4, q4)
		sol, status = lcp.solve_with_lemke_method(max_itr=20, verbose=True)
		self.assertEqual(status, 0)


if __name__ == '__main__':
    unittest.main()