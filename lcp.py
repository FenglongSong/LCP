import numpy as np
import matplotlib.pyplot as plt
from utils import *

class LinearComplementarityProblem:
    """
    Define the Linear Complementarity Problem (LCP).
    w - Mz = q,
    w, z >= 0,
    w^T z = 0.

    status:
        0: unsolved (initial state)
        1: feasible solution found
        2: ray termination
        3: infeasible
        4: max iteration reached

    """
    def __init__(self, M, q) -> None:
        self.M = M
        self.q = q
        self.n = q.shape[0]
        self.tableau = np.zeros([self.n, 2*self.n+2]) # [I, -M, -e, q]
        self.num_pivot_steps = 0
        self.basic_var_indecies = [*range(0, self.n)]
        self.status = 0
        self.lexico_tol = 1e-6
        self.zero_tol = 1e-10

    def initialize_tableau(self) -> None:
        n = self.n
        self.tableau[:, 0:n] = np.eye(self.n)
        self.tableau[:, n:2*n] = -self.M
        self.tableau[:, 2*n:2*n+1] = -np.ones([n,1])
        self.tableau[:, -1] = self.q
        
    def solve_with_lemke_method(self, max_itr:int=100, verbose:bool=False):
        """ Use Lemke's algorithm to compute a solution to the LCP.

        Returns:
        
        | exit_code | exit_string               | 
        -----------------------------------------
        |    0      | 'Solution Found'          |
        |    1      | 'Ray Termination'         |
        |    2      | 'Max Iterations Exceeded' |

        """

        if all(self.q >= 0):
            if verbose:
                print('Solution Found.')
            self.status = 0
            return np.concatenate((self.q, np.zeros(self.n))), self.status
        else:
            self.initialize_tableau()
            # first pivot (the z0 column)                
            pivot_row_index = np.argmin(self.q) # ! should be replaced to avoid cycle pivoting
            # ! what happens if there're two elements in q which are equivalent
            if verbose:
                print("Initial tablaeu is:\n", self.tableau)
                print("\n------------------ Iteration", 0, "------------------")
                print("Entering variable is: z0")
                print("Leaving variable is: ", self.find_variable_name(self.basic_var_indecies[pivot_row_index]))

            next_entering_var_index = self.find_complementary_index(self.basic_var_indecies[pivot_row_index]) # You have to find the complementary index before executing the pivot function, because pivot overwrite the basis_var_index
            self.pivot(pivot_row_index, 2*self.n)
            if verbose:
                print("The tableau after pivoting is: ")
                print(self.tableau)
            
            for i in range(1, max_itr):
                entering_var_index = next_entering_var_index
                pivot_col_index = entering_var_index

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

                # if all the values in ratios are infinties, we can say it's a ray termination?
                if all(ratios >= np.inf*np.ones(self.n)):
                    print("Ray termination! The algorithm is not able to solve the problem!")
                    self.status = 2
                    return 

                pivot_row_index = np.argmin(ratios)
                leaving_var_index = self.basic_var_indecies[pivot_row_index]

                if verbose:
                    print("\n------------------ Iteration", i, "------------------")
                    print("Entering variable is: ", self.find_variable_name(entering_var_index))
                    print("Leaving variable is: ", self.find_variable_name(self.basic_var_indecies[pivot_row_index]))

                self.pivot(pivot_row_index, entering_var_index)

                if verbose:
                    print("Current tableau after pivoting is: ")
                    print(self.tableau)

                if leaving_var_index == 2*self.n: # z0 leaves, solution found
                    if verbose:
                        print("\nFeasible solution found!")
                        print("basis is: ", self.basic_var_indecies)
                    break

                # compute the complementary index
                next_entering_var_index = self.find_complementary_index(leaving_var_index)

        # Exceed max iteration
        if i == max_itr-1:
            print("Max Iterations Exceeded.")
            self.status = 2
            return np.inf*np.ones(2*self.n), self.status
        

        q = self.tableau[:,-1]
        wz = np.zeros(2*self.n)
        for i in range(self.n):
            wz[self.basic_var_indecies[i]] = q[i]
        self.status = 0
        return wz, self.status

    def lexico_minimum_ratio_test(self, pivot_col_index) -> int:
        ''' Find the row index of the lexico minimum of tableau. 
        In non-degenerate case, just do minimum ratio test. 
        Lexico minimum ratio test is only carried out in degenerate case.
        '''

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
        
        
        # Test if there are multiple minimums in ratios (degenrate in q)
        min_ratios = np.min(ratios)
        min_count = np.count_nonzero(ratios == min_ratios)
        if min_count == 1:
            return np.argmin(ratios)
        else:
            tableau_basis_column = self.tableau[:, self.basic_var_indecies]
            beta = np.linalg.inv(tableau_basis_column)
            q_bar = np.reshape(beta @ q, (self.n,1))

            q_bar_and_beta = np.hstack((q_bar, beta)) # [beta*q, beta]
            c_bar_j = beta @ self.tableau[:, pivot_col_index]
            positive_c_bar_j_index = list(np.where(c_bar_j > self.zero_tol)[0])
            return lexico_argmin(q_bar_and_beta[positive_c_bar_j_index, :])


    
    def find_complementary_index(self, i:int) -> int:
        if i < 0 or i >= 2*self.n:
            raise ValueError("No complementary index")
        if i <= self.n-1:
            return i + self.n
        else:
            return i - self.n
        
    def find_variable_name(self, i:int) -> str:
        if i < 0 or i > 2*self.n:
            raise RuntimeError("Out of bound")
        if i <= self.n-1:
            return 'w'+str(i+1)
        elif i <= 2*self.n-1:
            return 'z'+str(i+1-self.n)
        else:
            return 'z0'

    def pivot(self, pivot_row_index, pivot_col_index) -> None:
        '''
        Perform pivoting at a given column (the entering variable) and a given row
        '''
        self.tableau[pivot_row_index, :] /= self.tableau[pivot_row_index, pivot_col_index]
        for i in range(self.n):
            if i != pivot_row_index:
                self.tableau[i, :] -= self.tableau[i, pivot_col_index] * self.tableau[pivot_row_index, :] 
        self.basic_var_indecies[pivot_row_index] = pivot_col_index