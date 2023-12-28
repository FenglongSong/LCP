import numpy as np
import matplotlib.pyplot as plt
import sys
'''
w - Mz = q,
w, z >= 0,
w^T z = 0.

status:
    0: unsolved (initial state)
    1: feasible solution found
    2: ray termination
    3: infeasible
    4: max iteration reached

'''

class LinearComplementarityProblem:
    def __init__(self, M, q) -> None:
        self.M = M
        self.q = q
        self.n = q.shape[0]
        self.tableau = np.zeros([self.n, 2*self.n+2]) # [I, -M, -e, q]
        self.num_pivot_steps = 0
        self.basic_var_indecies = [*range(0, self.n)]
        self.status = 0
        self.lexico_tol = 1e-6

    def initialize_tableau(self) -> None:
        n = self.n
        self.tableau[:, 0:n] = np.eye(self.n)
        self.tableau[:, n:2*n] = -self.M
        self.tableau[:, 2*n:2*n+1] = -np.ones([n,1])
        self.tableau[:, -1] = self.q
        
    def solve_with_lemke_method(self, max_itr:int=100, verbose:bool=False):
        if all(self.q >= 0):
            print("LCP solved!")
            return self.q, np.zeros((self.n,1))
        else:
            self.initialize_tableau()
            # first pivot (the z0 column)                
            pivot_row_index = np.argmin(self.q) # ! should be replaced to avoid cycle pivoting
            # ! what happens if there're two elements in q which are equivalent
            if verbose:
                print("Initial tablaeu is:\n", self.tableau)
                print("\n------------------ Iteration", 0, "------------------")
                print("Leaving variable is: ", self.find_variable_name(self.basic_var_indecies[pivot_row_index]))
                print("Entering variable is: z0")

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
                    if abs(self.tableau[j, pivot_col_index]) < sys.float_info.epsilon: # ! if the value=0, not sure if this is a good way
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
                    print("Leaving variable is: ", self.find_variable_name(self.basic_var_indecies[pivot_row_index]))
                    print("Entering variable is: ", self.find_variable_name(entering_var_index))

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

    
    def lexico_argmin(self) -> int:
        return NotImplementedError
    
    def find_complementary_index(self, i:int) -> int:
        if i < 0 or i >= 2*self.n:
            raise RuntimeError("No complementary index")
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