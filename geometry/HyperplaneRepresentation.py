import numpy as np
import casadi as ca


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
		self.H = np.hstack((A, np.reshape(b, (self.m, 1))))

	def remove_redundant_constraints(self) -> None:
		""" Remove all the redundant constraints.
		
		Reference: Section 2.20 of https://people.inf.ethz.ch/fukudak/Doc_pub/polyfaq220115c.pdf
		
		We want to test whether the subsystem of m-1 inequalities Cx \leq d implies the other one inequality s x \leq t. 
		If so, the inequality s x \leq t is redundant and can be removed from the system. A linear programming (LP) 
		formulation of this checking is rather straightforward:

		f^* = max s x
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

			opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.tol':1e-8}
			opti.solver('ipopt', opts)

			sol = opti.solve()
			if not sol.stats()['success']:
				raise RuntimeError("Ipopt failure.")
			
			x_opt = np.reshape(sol.value(x), (self.n,))
			f_opt = s @ x_opt
			if f_opt <= t + 1e-8: # redundant case:
				# Add a small number to avoid numerical issue. Otherwise may cause issue
				# when a redundant constraint is very close to a vertix.
				self.A = C
				self.b = d
				self.m -= 1
			else:                 # irredundant case:
				i += 1

			if i == self.m:
				break

			# A very naive way to avoid dead cycle
			if i >= 5000:
				raise RuntimeError("Dead cycle.")
				
		self.H = np.hstack((self.A, np.reshape(self.b, (self.m, 1))))