import numpy as np
from matplotlib import pyplot as plt
import pde
import time

X_SIZE = 20
Y_SIZE = 10
Z_SIZE = 10

class CustomPDE(pde.PDEBase):
    def __init__(self, bc, local_shape):
        self.bc = bc
        self.local_shape = local_shape
        self.vector_array_local = np.zeros((3, *local_shape))

        # Assuming each process gets a part of the X-dimension, center filling
        local_x_center = local_shape[0] // 2
        self.vector_array_local[0, max(0, local_x_center - 1):local_x_center + 1, :, :] = 1
        self.vector_array_local = np.ascontiguousarray(self.vector_array_local, dtype=np.float64)
        
        self.shear_viscosity = 0.1
        self.bulk_viscosity = 0.1

        # Operators initialized to None and will be defined per subgrid
        self.laplace_u = None
        self.divergence_u = None
        self.gradient_u = None
        self.gradient_u_2 = None

    def evolution_rate(self, state, t=0):
        # Calculate the local laplacian and other operators on the local subgrid
        laplacian = state.laplace(bc=self.bc)
        f_u = state.dot(state.gradient(bc=self.bc)) - self.shear_viscosity * laplacian \
              - (self.bulk_viscosity + self.shear_viscosity / 3) * state.divergence(bc=self.bc).gradient(bc=self.bc)

        # The local vector array is subtracted, this is now per subgrid (per process)
        ans = -f_u - 0.1 * self.vector_array_local
        return ans

# Set up the global grid
grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [X_SIZE, Y_SIZE, Z_SIZE], periodic=[False, False, False])
field = pde.VectorField(grid, data=0)
bc_x = ( {"value": 0})

# Decomposition of the grid, we are using 2 processes along X-axis
decomposition = [2, 1, 1]

# Each process gets part of the grid, so we must determine the local shape of the subgrid
local_shape = [X_SIZE // decomposition[0], Y_SIZE // decomposition[1], Z_SIZE // decomposition[2]]

# Initialize the PDE with localized shape
eq = CustomPDE(bc=[bc_x, bc_x, bc_x], local_shape=local_shape)

start_time = time.time()

# Use ExplicitMPISolver with the specified decomposition
result = eq.solve(field, t_range=1, dt=0.1, solver="explicit_mpi", decomposition=decomposition)

end_time = time.time()
print("Execution Time: ", end_time - start_time, " seconds")

# Disable plotting temporarily for debugging
# plot_2d_slice(result.data)
# result.to_scalar(scalar='norm').plot_interactive()

plt.show()
