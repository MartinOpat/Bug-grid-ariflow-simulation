import pde
import numpy as np
from pde.tools.numba import jit
import time as timer

def run_pde(eq):
    grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [20, 10, 10], periodic=[True, False, False])
    init = np.ones((20, 10, 10))
    # Fill the first 5x5 square with ones
    init[:5, :5, :5] = 2
    print(init)

    # field = pde.ScalarField.random_normal(grid, mean=0.5)
    field = pde.ScalarField(grid, init)
    field.plot_interactive()
    storage = pde.MemoryStorage()
    start_time = timer.time()
    result = eq.solve(field, t_range=20, dt=1e-5, tracker=["progress", storage.tracker(1)])
    end_time = timer.time()

    for time, field in storage.items():
        print(f"t={time}, field={field.magnitude}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    result.plot_interactive()


def simple_string_method():
    eq = pde.PDE({"u": "0.1*gradient_squared(u)"})
    run_pde(eq) 

class DiffusionPDE(pde.PDEBase):

    def evolution_rate(self, state, t=0):
        """Evaluate the right hand side of the evolution equation."""
        state_gradient_squared = state.gradient_squared(bc="auto_periodic_neumann")
        return state_gradient_squared * 0.1

def simple_class_method():
    eq = DiffusionPDE()
    run_pde(eq) 

class DiffusionJITPDE(pde.PDEBase):
    def evolution_rate(self, state, t=0): # NOT USED!!
        """Evaluate the right hand side of the evolution equation."""
        state_gradient_squared = state.gradient_squared(bc="auto_periodic_neumann")
        return state_gradient_squared * 0.01

    def _make_pde_rhs_numba(self, state):
        """ the numba-accelerated evolution equation """

        # create operators
        gradient_squared = state.grid.make_operator("gradient_squared", bc="auto_periodic_neumann")

        @jit
        def pde_rhs(state_data, t=0):
            """ compiled helper function evaluating right hand side """
            state_gradient_squared = gradient_squared(state_data)
            return state_gradient_squared * 0.01

        return pde_rhs

def jit_class_method():
    eq = DiffusionJITPDE()
    run_pde(eq) 

def built_in_method():
    eq = pde.DiffusionPDE(diffusivity=0.1)
    run_pde(eq)

# simple_string_method() # 18.37 seconds
# simple_class_method() # 251.9 seconds
jit_class_method() # 11.7 seconds # Without @jit 13.1 seconds
# built_in_method() # 14.8 seconds