# """
# Use multiprocessing via MPI
# ===========================

# Use multiple cores to solve a PDE. The implementation here uses the `Message Passing
# Interface (MPI) <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_, and the
# script thus needs to be run using :code:`mpiexec -n 2 python mpi_parallel_run.py`, where
# `2` denotes the number of cores used. Note that macOS might require an additional hint
# on how to connect the processes. The following line might work:
#     `mpiexec -n 2 -host localhost:2 python3 mpi_parallel_run.py`

# Such parallel simulations need extra care, since multiple instances of the same program
# are started. In particular, in the example below, the initial state is created on all
# cores. However, only the state of the first core will actually be used and distributed
# automatically by `py-pde`. Note that also only the first (or main) core will run the
# trackers and receive the result of the simulation. On all other cores, the simulation
# result will be `None`.
# """

# from pde import DiffusionPDE, ScalarField, UnitGrid

# grid = UnitGrid([64, 64])  # generate grid
# state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

# eq = DiffusionPDE(diffusivity=0.1)  # define the pde
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# if result is not None:  # check whether we are on the main core
#     result.plot()

###################################################################################

# from pde import DiffusionPDE, ScalarField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a larger grid
# grid = CartesianGrid([[0, 10], [0, 10]], [128, 128], periodic=[False, False])  # Larger Cartesian grid
# state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

# # Define the diffusion equation
# eq = DiffusionPDE(diffusivity=0.1)

# # Run the simulation using MPI solver
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Only the main core (rank 0) should plot the result
# if rank == 0 and result is not None:
#     result.plot()

###################################################################################

# import numpy as np
# from pde import DiffusionPDE, ScalarField, CartesianGrid
# from mpi4py import MPI



# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [128, 128], periodic=[False, False])

# # Generate a Gaussian blob as the initial condition
# state = ScalarField.from_expression(grid, "exp(-((x-5)**2 + (y-5)**2)/0.5)")

# # Define the diffusion equation
# eq = DiffusionPDE(diffusivity=0.1)

# # Run the simulation using MPI solver
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()

###################################################################################

# from pde import DiffusionPDE, ScalarField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [128, 128], periodic=[False, False])

# # Generate a Gaussian blob as the initial condition
# state = ScalarField.from_expression(grid, "exp(-((x-5)**2 + (y-5)**2)/0.5)")

# # Define the diffusion equation with Dirichlet boundary conditions
# eq = DiffusionPDE(diffusivity=0.1, bc={"value": 0})

# # Run the simulation using MPI solver
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()


###################################################################################


# from pde import PDEBase, ScalarField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [128, 128], periodic=[False, False])

# # Generate a Gaussian blob as the initial condition
# state = ScalarField.from_expression(grid, "exp(-((x-5)**2 + (y-5)**2)/0.5)")

# # Define a custom PDE class
# class CustomPDE(PDEBase):
#     def evolution_rate(self, state, t=0):
#         laplace = state.laplace(bc={"value": 0})  # Dirichlet boundary condition
#         return -0.1 * laplace  # Simple diffusion-like equation

# # Instantiate and run the custom PDE
# eq = CustomPDE()
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()

###################################################################################

# from pde import PDEBase, ScalarField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [128, 128], periodic=[False, False])

# # Generate a Gaussian blob as the initial condition
# state = ScalarField.from_expression(grid, "exp(-((x-5)**2 + (y-5)**2)/0.5)")

# # Define a custom PDE class with variable diffusivity and an external source
# class CustomPDEVariableDiffusivity(PDEBase):
#     def evolution_rate(self, state, t=0):
#         laplace = state.laplace(bc={"value": 0})  # Dirichlet boundary condition
        
#         # Define variable diffusivity as a function of position
#         diffusivity = ScalarField.from_expression(state.grid, "0.1 + 0.05 * exp(-((x-5)**2 + (y-5)**2))")
        
#         # External source term
#         source = ScalarField.from_expression(state.grid, "0.05 * exp(-((x-5)**2 + (y-5)**2)/0.5)")
        
#         # Apply variable diffusivity
#         return -diffusivity * laplace + source  # Diffusion equation with variable diffusivity and external source

# # Instantiate and run the custom PDE with variable diffusivity
# eq = CustomPDEVariableDiffusivity()
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()

###################################################################################

# import numpy as np
# from pde import PDEBase, VectorField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [64, 64], periodic=[False, False])

# # Create a vector field with some initial data (e.g., a vortex-like pattern)
# state = VectorField.from_expression(grid, ["sin(pi*x)*cos(pi*y)", "-cos(pi*x)*sin(pi*y)"])

# # Define a custom PDE class with vector fields and dynamically adjusted boundary mask
# class CustomPDEWithVectorField(PDEBase):
#     def __init__(self, bc):
#         self.bc = bc

#     def evolution_rate(self, state, t=0):
#         laplacian = state.laplace(bc=self.bc)  # Laplacian of the vector field

#         # Create a dynamic boundary mask based on the local grid coordinates
#         x, y = state.grid.cell_coords[..., 0], state.grid.cell_coords[..., 1]
#         boundary_mask = (x < 2) | (x > 8)  # Block evolution only near the edges
        
#         # Ensure that self.vector_array matches the shape of state.data
#         if not hasattr(self, 'vector_array') or self.vector_array.shape != state.data.shape:
#             self.vector_array = np.zeros(state.data.shape)
#             self.vector_array[0, state.data.shape[1]//2, :] = 1  # Adjust for the local subgrid size

#         # Restrict evolution to regions outside the boundary mask
#         result = -laplacian + self.vector_array
#         result.data[..., boundary_mask] = 0  # Zero out the evolution in the masked regions

#         return result

# # Instantiate and run the custom PDE with vector fields and boundary mask
# bc = {"value": 0}  # Dirichlet boundary condition
# eq = CustomPDEWithVectorField(bc)
# result = eq.solve(state, t_range=10, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()


###################################################################################


# import numpy as np
# from pde import PDEBase, VectorField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Create a grid
# grid = CartesianGrid([[0, 10], [0, 10]], [64, 64], periodic=[False, False])

# # Create a vector field with some initial data (e.g., a vortex-like pattern)
# state = VectorField.from_expression(grid, ["sin(pi*x)*cos(pi*y)", "-cos(pi*x)*sin(pi*y)"])

# # Define a custom PDE class with shear and bulk viscosity, vector fields, and boundary mask
# class CustomPDEWithViscosity(PDEBase):
#     def __init__(self, bc, shear_viscosity=0.1, bulk_viscosity=0.1):
#         self.bc = bc
#         self.shear_viscosity = shear_viscosity
#         self.bulk_viscosity = bulk_viscosity

#     def evolution_rate(self, state, t=0):
#         # Laplacian term
#         laplacian = state.laplace(bc=self.bc)

#         # Compute gradients and divergence for viscosity terms
#         gradient = state.gradient(bc=self.bc)
#         divergence = state.divergence(bc=self.bc)

#         # Shear and bulk viscosity terms
#         viscosity_term = (
#             -self.shear_viscosity * laplacian
#             - (self.bulk_viscosity + self.shear_viscosity / 3) * divergence.gradient(bc=self.bc)
#         )

#         # External vector array for forcing
#         if not hasattr(self, 'vector_array') or self.vector_array.shape != state.data.shape:
#             self.vector_array = np.zeros(state.data.shape)
#             self.vector_array[0, state.data.shape[1] // 2, :] = 1  # Central forcing along x-axis

#         # Final evolution rate with viscosity and external force
#         result = state.dot(gradient) + viscosity_term + self.vector_array

#         # Apply boundary mask (blocking evolution near the edges)
#         x, y = state.grid.cell_coords[..., 0], state.grid.cell_coords[..., 1]
#         boundary_mask = (x < 2) | (x > 8)
#         result.data[..., boundary_mask] = 0  # Zero out evolution in boundary masked regions

#         return result

# # Instantiate and run the custom PDE with viscosity
# bc = {"value": 0}  # Dirichlet boundary condition
# eq = CustomPDEWithViscosity(bc, shear_viscosity=0.1, bulk_viscosity=0.1)
# result = eq.solve(state, t_range=1, dt=0.1, solver="explicit_mpi")

# # Plot result on rank 0
# if rank == 0 and result is not None:
#     result.plot()


###################################################################################

# import numpy as np
# from pde import PDEBase, VectorField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Debug: Print MPI initialization details
# if rank == 0:
#     print(f"MPI initialized with {size} processes")

# # Create a 3D grid
# grid = CartesianGrid([[0, 10], [0, 10], [0, 10]], [32, 32, 32], periodic=[False, False, False])

# # Debug: Print the shape of the grid for each process
# print(f"Process {rank}: Grid shape {grid.shape}")

# # Create a 3D vector field with initial data
# state = VectorField.from_expression(grid, ["sin(pi*x)*cos(pi*y)*cos(pi*z)", "-cos(pi*x)*sin(pi*y)*cos(pi*z)", "cos(pi*x)*cos(pi*y)*sin(pi*z)"])

# # Define a custom 3D PDE class with a very simple evolution (constant array)
# class CustomPDETest(PDEBase):
#     def __init__(self, bc):
#         self.bc = bc

#     def evolution_rate(self, state, t=0):
#         # Simple evolution with no computation, just return a constant array for testing
#         evolution = VectorField(state.grid, data=0.01 * np.ones(state.data.shape))
        
#         # Debug: Print information during evolution rate computation
#         print(f"Process {rank}: Computing evolution rate at t={t}")
        
#         return evolution

# # Instantiate and run the custom 3D PDE with MPI
# bc = {"value": 0}  # Dirichlet boundary condition
# eq = CustomPDETest(bc)

# # Debug: Print before the solve
# if rank == 0:
#     print(f"Starting simulation with MPI...")

# # Use the MPI solver (explicit_mpi)
# result = eq.solve(state, t_range=1, dt=0.001, solver="explicit_mpi")

# # Debug: Print after the solve
# if rank == 0:
#     print(f"Simulation complete")

# # Plot result on rank 0 (only a 2D slice for visualization)
# if rank == 0 and result is not None:
#     result.to_scalar('norm').plot()

###################################################################################

import numpy as np
from pde import PDEBase, VectorField, CartesianGrid
from mpi4py import MPI

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Debug: Print MPI initialization details
if rank == 0:
    print(f"MPI initialized with {size} processes")

# Create a smaller 3D grid
grid = CartesianGrid([[0, 10], [0, 10], [0, 10]], [5, 81, 81], periodic=[False, False, False])

# Debug: Print the shape of the grid for each process
print(f"Process {rank}: Grid shape {grid.shape}")

# Create a 3D vector field with initial data
state = VectorField.from_expression(grid, ["sin(pi*x)*cos(pi*y)*cos(pi*z)", "-cos(pi*x)*sin(pi*y)*cos(pi*z)", "cos(pi*x)*cos(pi*y)*sin(pi*z)"])

# Define a custom 3D PDE class that includes Laplacian computation
class CustomPDEWithLaplacian(PDEBase):
    def __init__(self, bc):
        self.bc = bc

    def evolution_rate(self, state, t=0):
        # Laplacian term for diffusion-like evolution
        laplacian = state.laplace(bc=self.bc)
        
        # Debug: Print part of the Laplacian data to see if it's being computed
        print(f"Process {rank}: Laplacian at t={t}, sum={np.sum(laplacian.data)}")
        
        return laplacian

# Instantiate and run the custom 3D PDE with Laplacian
bc = {"value": 0}  # Dirichlet boundary condition
eq = CustomPDEWithLaplacian(bc)

# Debug: Print before the solve
if rank == 0:
    print(f"Starting simulation with MPI (Laplacian)...")

# Use the MPI solver (explicit_mpi)
result = eq.solve(state, t_range=1, dt=0.001, solver="explicit_mpi")

# Debug: Print after the solve
if rank == 0:
    print(f"Simulation complete")

# Plot result on rank 0 (only a 2D slice for visualization)
if rank == 0 and result is not None:
    result.to_scalar('norm').plot()

###################################################################################
# Suspicously stops working after adding laplacian + large field + n > 1

# import numpy as np
# from pde import PDEBase, VectorField, CartesianGrid
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Debug: Print MPI initialization details
# if rank == 0:
#     print(f"MPI initialized with {size} processes")

# # Reduce grid size to [24, 24, 24] for better performance
# grid = CartesianGrid([[0, 10], [0, 10], [0, 10]], [24, 24, 24], periodic=[False, False, False])

# # Debug: Print the shape of the grid for each process
# print(f"Process {rank}: Grid shape {grid.shape}")

# # Create a 3D vector field with initial data
# state = VectorField.from_expression(grid, ["sin(pi*x)*cos(pi*y)*cos(pi*z)", "-cos(pi*x)*sin(pi*y)*cos(pi*z)", "cos(pi*x)*cos(pi*y)*sin(pi*z)"])

# # Define a custom 3D PDE class that includes Laplacian computation
# class CustomPDEWithLaplacian(PDEBase):
#     def __init__(self, bc):
#         self.bc = bc

#     def evolution_rate(self, state, t=0):
#         # Laplacian term for diffusion-like evolution
#         laplacian = state.laplace(bc=self.bc)
        
#         # Debug: Print part of the Laplacian data to see if it's being computed
#         print(f"Process {rank}: Laplacian at t={t}, sum={np.sum(laplacian.data)}")
        
#         return laplacian

# # Instantiate and run the custom 3D PDE with Laplacian
# bc = {"value": 0}  # Dirichlet boundary condition
# eq = CustomPDEWithLaplacian(bc)

# # Debug: Print before the solve
# if rank == 0:
#     print(f"Starting simulation with MPI (Laplacian) with adaptive time stepping...")

# # Use the MPI solver with adaptive time stepping
# result = eq.solve(state, t_range=1, solver="explicit_mpi", adaptive=True)

# # Debug: Print after the solve
# if rank == 0:
#     print(f"Simulation complete")

# # Plot result on rank 0 (only a 2D slice for visualization)
# if rank == 0 and result is not None:
#     result.to_scalar('norm').plot()

##################################################################################

# import numpy as np
# from pde import ScalarField, CartesianGrid, DiffusionPDE
# from mpi4py import MPI

# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Print MPI initialization
# if rank == 0:
#     print(f"MPI initialized with {size} processes")

# # Create a 3D grid (32x32x32) on each process
# grid = CartesianGrid([[0, 10], [0, 10], [0, 10]], [24, 24, 24], periodic=[False, False, False])

# # Create an initial condition: a 3D scalar field (temperature distribution)
# initial_state = ScalarField.random_uniform(grid)

# # Define the diffusion PDE with a constant diffusivity (representing the heat equation)
# alpha = 0.1  # Thermal diffusivity
# eq = DiffusionPDE(diffusivity=alpha)

# # Debug: Print the grid shape for each process
# print(f"Process {rank}: Grid shape {grid.shape}")

# # Solve the PDE using MPI
# if rank == 0:
#     print("Starting 3D Heat Equation simulation with MPI...")

# # result = eq.solve(initial_state, t_range=1, dt=0.01, solver="explicit_mpi", decomposition=[2, 2, 1])
# result = eq.solve(initial_state, t_range=1, dt=0.01)

# # Print success message after completion
# if rank == 0:
#     print("3D Heat Equation simulation completed.")

# # Optionally, plot the result (only from process 0)
# if rank == 0:
#     result.plot()


###################################################################################



# import numpy as np
# from mpi4py import MPI
# from scipy.ndimage import laplace

# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Parameters for the heat equation
# alpha = 0.01  # diffusivity
# nx, ny, nz = 64, 64, 64  # grid size
# dx = dy = dz = 1.0  # grid spacing
# dt = 0.01  # time step
# t_final = 0.1  # final time

# # Divide the domain along the x-direction
# local_nx = nx // size
# if rank == 0:
#     print(f"Grid size: {nx}x{ny}x{nz}, each process gets {local_nx}x{ny}x{nz}")

# # Initialize the temperature field for each process
# local_u = np.zeros((local_nx + 2, ny, nz))  # Add ghost cells for boundary exchange
# local_u_new = np.zeros_like(local_u)

# # Set an initial condition (Gaussian)
# x = np.linspace(rank * local_nx * dx, (rank + 1) * local_nx * dx, local_nx + 2)
# y = np.linspace(0, ny * dy, ny)
# z = np.linspace(0, nz * dz, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
# local_u[:] = np.exp(-((X - 32) ** 2 + (Y - 32) ** 2 + (Z - 32) ** 2) / 100.0)

# # MPI boundary exchange function
# def exchange_boundaries(local_u, comm, rank, size):
#     # Send to the left, receive from the right
#     if rank > 0:
#         comm.Sendrecv(local_u[1, :, :], dest=rank-1, sendtag=11,
#                       recvbuf=local_u[0, :, :], source=rank-1, recvtag=22)
#     if rank < size - 1:
#         comm.Sendrecv(local_u[-2, :, :], dest=rank+1, sendtag=22,
#                       recvbuf=local_u[-1, :, :], source=rank+1, recvtag=11)

# # Time integration loop
# t = 0.0
# while t < t_final:
#     # Exchange boundaries between processes
#     exchange_boundaries(local_u, comm, rank, size)
    
#     # Apply the Laplacian operator (finite difference)
#     lap_u = laplace(local_u[1:-1, :, :], mode="constant")
    
#     # Update temperature field using explicit time-stepping
#     local_u_new[1:-1, :, :] = local_u[1:-1, :, :] + alpha * dt * lap_u
    
#     # Swap references for next iteration
#     local_u, local_u_new = local_u_new, local_u
    
#     # Advance time
#     t += dt

#     if rank == 0:
#         print(f"Time {t:.4f} / {t_final}")

# # Gather the result from all processes (just for visualization or saving)
# global_u = None
# if rank == 0:
#     global_u = np.zeros((nx, ny, nz))

# comm.Gather(local_u[1:-1, :, :], global_u, root=0)

# # Output final state (only by rank 0)
# if rank == 0:
#     print("Final state gathered successfully.")

# # You can add visualization or saving functionality for global_u if needed


###################################################################################

# import numpy as np
# from mpi4py import MPI
# from scipy.ndimage import laplace

# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Parameters for the PDE
# alpha = 0.01  # diffusivity (heat term)
# beta = 0.1  # advection coefficient (velocity field strength)
# nx, ny, nz = 64, 64, 64  # grid size
# dx = dy = dz = 1.0  # grid spacing
# dt = 0.01  # time step
# t_final = 0.1  # final time

# # Divide the domain along the x-direction
# local_nx = nx // size
# if rank == 0:
#     print(f"Grid size: {nx}x{ny}x{nz}, each process gets {local_nx}x{ny}x{nz}")

# # Initialize the fields for each process
# local_u = np.zeros((local_nx + 2, ny, nz))  # Add ghost cells for boundary exchange
# local_u_new = np.zeros_like(local_u)

# # Initial condition: Gaussian bump in the middle
# x = np.linspace(rank * local_nx * dx, (rank + 1) * local_nx * dx, local_nx + 2)
# y = np.linspace(0, ny * dy, ny)
# z = np.linspace(0, nz * dz, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
# local_u[:] = np.exp(-((X - 32) ** 2 + (Y - 32) ** 2 + (Z - 32) ** 2) / 100.0)

# # Velocity field for advection (constant in this case)
# velocity = np.array([1.0, 0.0, 0.0])  # Advection in the x-direction

# # MPI boundary exchange function
# def exchange_boundaries(local_u, comm, rank, size):
#     # Send to the left, receive from the right
#     if rank > 0:
#         comm.Sendrecv(local_u[1, :, :], dest=rank-1, sendtag=11,
#                       recvbuf=local_u[0, :, :], source=rank-1, recvtag=22)
#     if rank < size - 1:
#         comm.Sendrecv(local_u[-2, :, :], dest=rank+1, sendtag=22,
#                       recvbuf=local_u[-1, :, :], source=rank+1, recvtag=11)

# # Apply Neumann boundary condition (zero gradient) on the y and z boundaries
# def apply_neumann_bc(local_u):
#     local_u[:, 0, :] = local_u[:, 1, :]  # Neumann on y-min
#     local_u[:, -1, :] = local_u[:, -2, :]  # Neumann on y-max
#     local_u[:, :, 0] = local_u[:, :, 1]  # Neumann on z-min
#     local_u[:, :, -1] = local_u[:, :, -2]  # Neumann on z-max

# # Time integration loop
# t = 0.0
# while t < t_final:
#     # Exchange boundaries between processes
#     exchange_boundaries(local_u, comm, rank, size)
    
#     # Apply Neumann boundary conditions
#     apply_neumann_bc(local_u)

#     # Calculate Laplacian for diffusion
#     lap_u = laplace(local_u[1:-1, :, :], mode="constant")
    
#     # Calculate advection term (using upwind scheme for simplicity)
#     adv_u = -beta * (local_u[1:-1, :, :] - local_u[:-2, :, :]) / dx
    
#     # Update the field (advection-diffusion equation)
#     local_u_new[1:-1, :, :] = local_u[1:-1, :, :] + dt * (alpha * lap_u + adv_u)
    
#     # Swap references for next iteration
#     local_u, local_u_new = local_u_new, local_u
    
#     # Advance time
#     t += dt

#     if rank == 0:
#         print(f"Time {t:.4f} / {t_final}")

# # Gather the result from all processes (just for visualization or saving)
# global_u = None
# if rank == 0:
#     global_u = np.zeros((nx, ny, nz))

# comm.Gather(local_u[1:-1, :, :], global_u, root=0)

# # Output final state (only by rank 0)
# if rank == 0:
#     print("Final state gathered successfully.")

# # You can add visualization or saving functionality for global_u if needed




###################################################################################

# import numpy as np
# from mpi4py import MPI
# from scipy.ndimage import laplace

# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Parameters for the PDE
# alpha = 0.01  # diffusivity
# beta = 0.1  # advection coefficient
# nx, ny, nz = 64, 64, 64  # grid size
# dx = dy = dz = 1.0  # grid spacing
# dt = 0.01  # time step
# t_final = 0.5  # final time

# # Source term (optional, can be removed or modified)
# def source_term(x, y, z):
#     return np.sin(x) * np.cos(y) * np.exp(-z)

# # Divide the domain along the x-direction
# local_nx = nx // size
# if rank == 0:
#     print(f"Grid size: {nx}x{ny}x{nz}, each process gets {local_nx}x{ny}x{nz}")

# # Initialize the fields for each process
# local_u = np.zeros((local_nx + 2, ny, nz))  # Add ghost cells for boundary exchange
# local_u_new = np.zeros_like(local_u)

# # Initial condition: Gaussian bump in the middle
# x = np.linspace(rank * local_nx * dx, (rank + 1) * local_nx * dx, local_nx + 2)
# y = np.linspace(0, ny * dy, ny)
# z = np.linspace(0, nz * dz, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
# local_u[:] = np.exp(-((X - 32) ** 2 + (Y - 32) ** 2 + (Z - 32) ** 2) / 100.0)

# # Velocity field for advection (constant)
# velocity = np.array([1.0, 0.0, 0.0])

# # MPI boundary exchange function
# def exchange_boundaries(local_u, comm, rank, size):
#     if rank > 0:
#         comm.Sendrecv(local_u[1, :, :], dest=rank-1, sendtag=11,
#                       recvbuf=local_u[0, :, :], source=rank-1, recvtag=22)
#     if rank < size - 1:
#         comm.Sendrecv(local_u[-2, :, :], dest=rank+1, sendtag=22,
#                       recvbuf=local_u[-1, :, :], source=rank+1, recvtag=11)

# # Apply Neumann boundary conditions on y- and z-boundaries
# def apply_neumann_bc(local_u):
#     local_u[:, 0, :] = local_u[:, 1, :]  # Neumann on y-min
#     local_u[:, -1, :] = local_u[:, -2, :]  # Neumann on y-max
#     local_u[:, :, 0] = local_u[:, :, 1]  # Neumann on z-min
#     local_u[:, :, -1] = local_u[:, :, -2]  # Neumann on z-max

# # Time integration loop
# t = 0.0
# while t < t_final:
#     # Exchange boundaries between processes
#     exchange_boundaries(local_u, comm, rank, size)
    
#     # Apply Neumann boundary conditions
#     apply_neumann_bc(local_u)

#     # Calculate Laplacian for diffusion
#     lap_u = laplace(local_u[1:-1, :, :], mode="constant")
    
#     # Calculate advection term (upwind scheme)
#     adv_u = -beta * (local_u[1:-1, :, :] - local_u[:-2, :, :]) / dx
    
#     # Calculate source term
#     src_u = source_term(X[1:-1, :, :], Y[1:-1, :, :], Z[1:-1, :, :])

#     # Update the field (advection-diffusion equation with source)
#     local_u_new[1:-1, :, :] = local_u[1:-1, :, :] + dt * (alpha * lap_u + adv_u + src_u)
    
#     # Swap references for next iteration
#     local_u, local_u_new = local_u_new, local_u
    
#     # Advance time
#     t += dt

#     if rank == 0:
#         print(f"Time {t:.4f} / {t_final}")

# # Gather the result from all processes (just for visualization or saving)
# global_u = None
# if rank == 0:
#     global_u = np.zeros((nx, ny, nz))

# comm.Gather(local_u[1:-1, :, :], global_u, root=0)

# # Output final state (only by rank 0)
# if rank == 0:
#     print("Final state gathered successfully.")
