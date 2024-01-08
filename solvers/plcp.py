import numpy as np
import casadi as ca

from geometry.Polyhedron import Polyhedron
from geometry.HyperplaneRepresentation import HyperplaneRepresentation
from utils import *
from solvers.lcp import *


class MultiparametricLinearComplementarityProblem:
    """
    Define the Parametric Linear Complementarity Problem (pLCP).

    w - M*z = q + Q*theta,
    w, z >= 0,
    w^T z = 0,
    A_theta * theta <= b_theta
    where Q\in R^{n\times d} is a real matrix of rank d.

    """

    def __init__(self, M: np.ndarray, q: np.ndarray, Q: np.ndarray, 
                 A_theta: np.ndarray, b_theta: np.ndarray) -> None:
        
        self.M = M
        self.q = q
        self.Q = Q
        self.n = M.shape[0]
        self.d = Q.shape[1]

        self.A = np.hstack((np.eye(self.n), -self.M))  # the value should not be changed

        # define the parameter space
        self.A_theta = A_theta
        self.b_theta = b_theta

        self.status = 0
        self.lexico_tol = 1e-6
        self.zero_tol = 1e-8
        self.max_itr = 1000

        self.basic_var_indices = [*range(0, self.n)]
        self.tableau = np.hstack((np.eye(self.n), -self.M, np.zeros((self.n,2))))  # used in the pivot steps in computing adjacent regions

        self.unexplored_bases = set()
        self.discovered_bases = set()


    def solve(self) -> None:
        """
        Solve the Multiparametric Linear Complementarity Problem.
        """

        # Initialisation, find a feasible complementary basis B0
        theta_init = self.compute_initial_feasible_param()
        lcp = LinearComplementarityProblem(self.M, self.q+self.Q@theta_init)
        _, status = lcp.solve_with_lemke_method(verbose=True)
        if status:
            raise RuntimeError("Fail to solve LCP with initial parameter theta")
        B0 = Basis(lcp.basic_var_indices, self.n)

        self.discovered_bases.add(B0)
        self.unexplored_bases.add(B0)


        while len(self.unexplored_bases):

            # select any basis from unexplored_bases and remove it
            basis_to_explore = list(self.unexplored_bases)[0] # ? should we change way to select rather than just select the first one?
            self.unexplored_bases.remove(basis_to_explore)

            # explore this basis
            region_to_explore = self.compute_critical_region(basis_to_explore, True)

            # for each facet of the closure of this region, find it's neighbors by pivoting
            number_of_facets = np.size(region_to_explore.b)

            for i in range(number_of_facets):
                adjacent_region_basis = self.compute_adjacent_critical_region(region_to_explore, i)

                # line 6 and 7 in Algorithm 1 in CNJ2006
                self.unexplored_bases = self.unexplored_bases.union({adjacent_region_basis} - self.discovered_bases)
                self.discovered_bases = self.discovered_bases.union({adjacent_region_basis})



    def compute_initial_feasible_param(self) -> np.ndarray:
        """ Find a parameter theta s.t the LCP (q+Q*theta, M) is feasible.

        Relax the complementarity constraints w^T z = 0 and solve the following problem:
        min 1^T * [\theta; x]
        s.t. A * x = q + Q * theta,
             x >= 0,
             A_theta * theta <= b_theta

        where x = [w; z], A = [I, -M].
        Q has a shape of (n, d)

        Returns:
            np.ndarray with a shape of (d,)
        """
        opti = ca.Opti()
        x = opti.variable(2*self.n)
        theta = opti.variable(self.d)
        opti.subject_to(x >= 0)
        opti.subject_to(self.A_theta @ theta <= self.b_theta)
        opti.subject_to(np.hstack((np.eye(self.n), -self.M)) @ x == self.q + self.Q @ theta)
        opti.minimize(ca.dot(np.ones(2*self.n), x) + ca.dot(np.ones(self.d), theta))
        
        options = {'ipopt.print_level':0, 'print_time':0, 'ipopt.tol':1e-8}
        opti.solver('ipopt', options)

        sol = opti.solve()
        if not sol.stats()['success']:
            raise RuntimeError("Ipopt failure.")
        
        theta_opt = sol.value(theta)
        return np.reshape(theta_opt, -1)


    # TODO: add unit test for this function
    def compute_critical_region(self, basis: Basis, remove_redundancy: bool = True) -> Polyhedron:
        """ Compute the closure of a full-dimension critical region given a basis.

        The closure of a full-dimension cirtical region is:
        R_B = {theta | beta * (Q theta + q) >= 0}, where beta is the inverse of A_{*,B}

        Args:
            basis: a basis, denoted as B.
            remove_redundancy: perform redundancy removal or not.
        
        Returns:
            A polyhedron, representing the closure of critical region.
        """
        beta = np.linalg.inv(self.A[:, basis.indices]) # TODO: how to represent the tableau and compute beta
        Hrep = HyperplaneRepresentation(-beta@self.Q, beta@self.q)
        if remove_redundancy:
            Hrep.remove_redundant_constraints()
        return Polyhedron(Hrep)
    
    
    def compute_adjacent_critical_region(self, region: Polyhedron,
                                        facet_index: int) -> Basis:
        """ Compute the adjacent region along the given facet via pivoting.

        Args:
            region: current region of which the adjacent region will be computed.
            facet_index: index of the facet to step over.
            basis: 


        """
        # get normalized facet equation gamma'*theta = n, where ||gamma||_2 = 1
        gamma = region.H[facet_index, 0:-1]
        b = region.H[facet_index, -1]
        b /= np.linalg.norm(gamma)
        gamma /= np.linalg.norm(gamma)

        # Elimination proceeds as follows:
        # g1*th1 + g2*th2 + ... + gmax*thp + ... + gd*thd = b
        
        # extract thp:
        # thp = 1/gmax * (b - g1*th1 -g2*th2 - ... - gd*thd)
        # thp = - 1/gmax * (g1*th1 +g2*th2 + ... + gd*thd) + b/gmax
        
        # put thp inside Q*th:
        # Q*th = [q11*th1 + q12*th2 + ... + q1p*thp + ... + q1d*thd; ...]
        #      = [(q11-qp*g1/gmax)*th1 + ... + (q1d-qp*gd/gmax)*thd + qp*b/gmax; ...] (except thp)
        
        # From the above equation we extract matrices such that Qnew*thf + qnew 
        # thf is theta except thp, Qnew corresponds to columns of Q that multiply thf
        # qnew is extracted and added to remaining term q
        # Qnew = Q(:,np) - Q(:,p)*gamma(np)'/gmax
        # qnew = Q(:,p)*b/gmax + q

        index_gamma_max = np.argmax(abs(gamma))
        gamma_max = gamma[index_gamma_max]

        # indices of all except gamma_max
        indices_except_gamma_max = [True] * self.d
        indices_except_gamma_max[index_gamma_max] = False

        # new LCP problem, including variable alpha for step size
        # w - M*z  = qnew + Qnew*thf + Q*gamma*alpha  + epsilon
        # [I -M -Q*gamma]*[w;z;alpha] = qnew + Qnew*thf + epsilon

        # ! Maybe we can focus on d=1 or d=2 as simple cases, because we can get ride of 
        # ! lexico-min test. Just make sure the whole logic works, and then go back to higher dim cases.
        if self.d == 1:
            # For parameter space of 1-dim, reduce to non-parametric LCP
            Q_hat = np.zeros((self.n, self.d))
        else:
            Q_hat = self.Q[:, indices_except_gamma_max] - self.Q[:, [index_gamma_max]] @ gamma[indices_except_gamma_max] / gamma_max

        q_hat = self.Q[:, index_gamma_max] * b/gamma_max + self.q
        self.tableau = np.hstack((np.eye(self.n), 
                                  -self.M, 
                                  -self.Q @ np.reshape(gamma, (-1,1)), 
                                  np.reshape(q_hat, (-1,1)) 
                                  ))

        # TODO: do we need to compute the set F_hat?

        basic_var_indices = list(range(self.n))

        # perform pivoting
        # choose alpha to be the entering variable
        entering_var_index = 2*self.n
        # ? Really? You have to find the complementary index before executing the pivot function, because pivot overwrite the basis_var_index
        leaving_var_index = self.lexico_minimum_ratio_test(entering_var_index)
        next_entering_var_index = self.find_complementary_index(basic_var_indices[leaving_var_index]) 
        
        self.pivot(leaving_var_index, entering_var_index)

        for i in range(self.max_itr):
            entering_var_index = next_entering_var_index
            pivot_row_index = self.lexico_minimum_ratio_test(entering_var_index)
            leaving_var_index = basic_var_indices[pivot_row_index]

            self.pivot(pivot_row_index, entering_var_index)

            # termination
            if leaving_var_index == 2*self.n: # z0 leaves, solution found
                break

            next_entering_var_index = self.find_complementary_index(leaving_var_index)

        return Basis(self.basic_var_indices, self.n)
    
    
    def find_complementary_index(self, i: int) -> int:
        if i < 0 or i >= 2*self.n:
            raise ValueError("No complementary index")
        if i <= self.n-1:
            return i + self.n
        else:
            return i - self.n
    

    # def lexico_minimum_ratio_test(self, pivot_col_index: int, F_hat: Polyhedron) -> int:
    def lexico_minimum_ratio_test(self, pivot_col_index: int) -> int:
        """
        Perform lexico minimum ratio test to determine the leaving variable.
        The index of leaving variable is the index of pivoting row.

        Reference: Section III.B.1.a of CNJ2006.
        make sure that: beta * (q_hat + Q_hat * theta_f + epsilon) >= 0, for all theta_f in F_hat
        i.e.,

        Args:
            pivot_col_index: the index of pivoting column (index of entering variable).
            F_hat: the parameter set of theta_f (refer to Section III.B.1 of CNJ2006) 

        """
        # If non-degenerate, just do minimum ratio test
        # determine the next leaving variable (choose row)
        ratios = np.ones(self.n)
        q = self.tableau[:, -1]
        for j in range(self.n):
            if abs(self.tableau[j, pivot_col_index]) < self.zero_tol:
                ratios[j] = np.inf
            else:
                ratios[j] = q[j] / self.tableau[j, pivot_col_index]

        # deal with the negative ratios
        for j in range(self.n):
            if ratios[j] < 0:
                ratios[j] = np.inf

        # TODO: haven't consider more complicated case.
        return np.argmin(ratios)


    def pivot(self, pivot_row_index, pivot_col_index) -> None:
        """ 
        Perform pivoting at a given column (the entering variable) 
        and a given row (the leaving variable).
        """

        self.tableau[pivot_row_index, :] /= self.tableau[pivot_row_index, pivot_col_index]
        for i in range(self.n):
            if i != pivot_row_index:
                self.tableau[i, :] -= self.tableau[i, pivot_col_index] * self.tableau[pivot_row_index, :] 
        self.basic_var_indices[pivot_row_index] = pivot_col_index


    def pivoting(self, aclfb: Basis, entering_index: int):
        """
        The pivot function in Algorithm 2 in CNJ2006.
        """
        beta = np.linalg.inv(self.A[:, aclfb.indices])

        # find the set Z
        # TODO:

        # find the set P
        # TODO:

        




