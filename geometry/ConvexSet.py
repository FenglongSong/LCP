from abc import ABC, abstractmethod 

class ConvexSet(ABC):
    """
    Define the convex set.
    """

    @abstractmethod
    def is_point_inside(self, point):
        """Check if a point is inside the convex set."""
        pass

    @abstractmethod
    def volume(self):
        """Calculate the volume of the convex set."""
        pass