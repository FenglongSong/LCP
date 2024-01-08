from geometry.HyperplaneRepresentation import HyperplaneRepresentation
from geometry.ConvexSet import ConvexSet


class Polyhedron(ConvexSet):
	"""
	Define the (convex) polyhedron.
	"""

	def __init__(self, reprenestation) -> None:
		if isinstance(reprenestation, HyperplaneRepresentation):
				self.Hrep = reprenestation
				self.dim = reprenestation.n
				self.H = reprenestation.H
				self.A = reprenestation.A
				self.b = reprenestation.b
		else:
				raise NotImplementedError