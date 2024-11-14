import numpy as np
from matplotlib import pyplot as plt
import pde
from pde.tools.numba import jit
from numba import prange
import time

X_SIZE = 100 # should be divisible by 2
Y_SIZE = 250
Z_SIZE = 1

class CustomPDE(pde.PDEBase):

    def __init__(self, bc, bc_density, boundary_mask, bc_vec):
        self.bc = bc
        self.boundary_mask = np.ascontiguousarray(boundary_mask, dtype=np.bool_)
        self.vector_array = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
        self.vector_array[0, -1 + X_SIZE//2:1 + X_SIZE//2, :, :] = 10
        self.vector_array = np.ascontiguousarray(self.vector_array, dtype=np.float64)
        self.dynamic_viscosity = 0.01  # 10^4*[SI]
        self.bulk_viscosity = 0.1  # 10^4*[SI]
        self.bc_vec = bc_vec
        self.RT = 300

        self.bc_density = bc_density

        self.laplace_u = None
        self.divergence_u = None
        self.gradient_u = None
        self.gradient_u_2 = None

    def evolution_rate(self, states, t=0):
        state, density = states

        # print(state.data[:, 0, Y_SIZE//2, Z_SIZE//2])

        # state.data[:, :, 0, :] = 0
        # state.data[:, :, -1, :] = 0
        # state.data[:, :, :, 0] = 0
        # state.data[:, :, :, -1] = 0
        # state.data[0, 0, :, :] = -v0
        state.data[0, -1, :, :] = -v0
        state.data[:, self.boundary_mask] = 0  # This sets the values to 0

        # Cap speed at \sqrt{3} v0
        state.data[0, :, :, :] = np.clip(state.data[0, :, :, :], -v0, v0)
        state.data[1, :, :, :] = np.clip(state.data[1, :, :, :], -v0, v0)
        state.data[2, :, :, :] = np.clip(state.data[2, :, :, :], -v0, v0)

        laplacian = state.laplace(bc=self.bc_vec)
        f_u = state.dot(state.gradient(bc=self.bc_vec)) - (self.dynamic_viscosity/density) * laplacian \
              - (self.bulk_viscosity/density + self.dynamic_viscosity / (3*density)) * state.divergence(bc=self.bc_vec).gradient(bc=self.bc)

        
        # pressure = density*self.RT
        # pressure =-0.5 * density * state.dot(state)
        pressure = np.zeros_like(density.data)
        pressure = pde.ScalarField(density.grid, data=pressure)
        # print(pressure.data[:2])
        ans = -f_u - pressure.gradient(bc=self.bc_density) / density.data[np.newaxis, :, :, :]
        # print(ans.data.any())
        ans.data[:, self.boundary_mask] = 0  # This sets the derivatives to 0

        density_derivative = np.zeros_like(density.data)
        density_derivative = pde.ScalarField(density.grid, data=density_derivative)

        return pde.FieldCollection([ans, density_derivative])

    # def _make_pde_rhs_numba(self, states): 
    #     state, state_density = states
    #     if self.laplace_u is None:
    #         self.laplace_u = state.grid.make_operator("vector_laplace", bc=self.bc_vec)
    #     if self.divergence_u is None:
    #         self.divergence_u = state.grid.make_operator("divergence", bc=self.bc_vec)
    #     if self.gradient_u is None:
    #         self.gradient_u = state.grid.make_operator("vector_gradient", bc=self.bc_vec)
    #     if self.gradient_u_2 is None:
    #         self.gradient_u_2 = state.grid.make_operator("gradient", bc=self.bc)
        
    #     laplace_u = self.laplace_u
    #     divergence_u = self.divergence_u
    #     gradient_u = self.gradient_u
    #     gradient_u_2 = self.gradient_u_2

    #     divergence_p_u = state_density.grid.make_operator("divergence", bc=self.bc_density)
    #     gradient_p_u = state_density.grid.make_operator("gradient", bc=self.bc_density)

    #     @jit(nopython=True, parallel=True)
    #     def convective_derivative(u, gradient_u_x, gradient_u_y, gradient_u_z):
    #         result = np.zeros_like(u)
    #         for i in prange(3):
    #             result[i] = u[0] * gradient_u_x[i] + u[1] * gradient_u_y[i] + u[2] * gradient_u_z[i]
    #         return result


    #     dynamic_viscosity = self.dynamic_viscosity
    #     bulk_viscosity = self.bulk_viscosity
    #     boundary_mask = self.boundary_mask
    #     RT = self.RT

    #     @jit(nopython=True, parallel=True)
    #     def apply_boundary_mask(vector_field, boundary_mask):
    #         for i in prange(X_SIZE):
    #             for j in range(Y_SIZE):
    #                 for k in range(Z_SIZE):
    #                     if boundary_mask[i][j][k]:
    #                         vector_field[0][i][j][k] = 0
    #                         vector_field[1][i][j][k] = 0
    #                         vector_field[2][i][j][k] = 0

    #     @jit(nopython=True, parallel=True)
    #     def apply_boundary_mask_scalar(scalar_field, boundary_mask):
    #         for i in prange(X_SIZE):
    #             for j in range(Y_SIZE):
    #                 for k in range(Z_SIZE):
    #                     if boundary_mask[i][j][k]:
    #                         scalar_field[i][j][k] = 0

        
    #     @jit(nopython=True, parallel=True)
    #     def multiply_scalar_vector(scalar, vector):
    #         result = np.zeros_like(vector)
    #         for i in prange(3):
    #             result[i] = scalar * vector[i]
    #         return result

    #     @jit(nopython=True, parallel=True)
    #     def multiply_scalar_vector_field(scalar_field, vector_field):
    #         result = np.zeros_like(vector_field, dtype=np.float64)
    #         for x in prange(X_SIZE):
    #             for y in range(Y_SIZE):
    #                 for z in range(Z_SIZE):
    #                     result[0][x][y][z] = vector_field[0][x][y][z]*scalar_field[x][y][z]
    #                     result[1][x][y][z] = vector_field[1][x][y][z]*scalar_field[x][y][z]
    #                     result[2][x][y][z] = vector_field[2][x][y][z]*scalar_field[x][y][z]

    #         return result


    #     @jit(nopython=True, parallel=True)
    #     def pde_rhs(state_datas, t=0):
    #         state_data = state_datas[0:3]
    #         state_density_data = state_datas[3]
    #         apply_boundary_mask(state_data, boundary_mask)


    #         # print(type(state_density_data))
    #         # state_data, state_density_data = state_datas
    #         state_lapacian = laplace_u(state_data)
    #         state_grad = gradient_u(state_data)
    #         state_grad2 = gradient_u_2(divergence_u(state_data))
    #         product_p_u = multiply_scalar_vector_field(state_density_data, state_data)
    #         state_divergence_p = divergence_p_u(product_p_u)

    #         shear_viscosity = dynamic_viscosity/state_density_data
    #         bulk_kinematic_viscosity = bulk_viscosity/state_density_data

    #         con_dev = convective_derivative(state_data, state_grad[0], state_grad[1], state_grad[2])

    #         shear_lap = multiply_scalar_vector(shear_viscosity, state_lapacian)

    #         vis_grad2 = multiply_scalar_vector(bulk_kinematic_viscosity + shear_viscosity / 3, state_grad2)

    #         f_u = con_dev - shear_lap - vis_grad2

    #         pressure = state_density_data*RT
    #         # ans = -f_u - pressure.gradient(bc=self.bc_density) / density.data[np.newaxis, :, :, :]
    #         ans = -f_u - multiply_scalar_vector_field(1/state_density_data, gradient_p_u(pressure))

    #         apply_boundary_mask(ans, boundary_mask)  # Moved at the begginning of the function

    #         density_t = -state_divergence_p
    #         apply_boundary_mask_scalar(density_t, boundary_mask)
    #         # return np.stack([ans, density_t])
    #         return np.concatenate((ans, density_t[np.newaxis,:,:,:]), axis=0)

    #     return pde_rhs

class LivePlotTracker2(pde.LivePlotTracker):
    grid_size = [X_SIZE, Y_SIZE]
    grid = pde.UnitGrid(grid_size)  
    z_slice = Z_SIZE // 2

    def initialize(self, state: pde.FieldBase, info = None) -> float:
        field_obj = pde.ScalarField(self.grid)
        return super().initialize(field_obj, info)

    def handle(self, state: pde.FieldBase, t: float) -> None:
        # sliced_values = state.data[3][:, :, self.z_slice]
        sliced_values = np.linalg.norm(state.data[:3], axis=0)[:, :, self.z_slice]
        field_obj = pde.ScalarField(self.grid, data=sliced_values)
        super().handle(field_obj, t)

def plot_2d_slice(vector_field):
    # Extract the u, v, w components
    u = vector_field[0]
    v = vector_field[1]
    w = vector_field[2]

    # Choose a specific Z slice (for example, the middle plane)
    z_slice = Z_SIZE // 2  # Taking the middle slice, but this can be any valid Z index

    # Slice the vector field at the chosen Z-plane
    u_slice = u[:, :, z_slice]
    v_slice = v[:, :, z_slice]
    w_slice = w[:, :, z_slice]  # We'll ignore this since it's a 2D plot

    # Create the grid corresponding to the X and Y dimensions
    x, y = np.meshgrid(np.linspace(0, 10, num=X_SIZE), np.linspace(0,5, num=Y_SIZE), indexing='ij')

    # Compute the magnitude for the 2D vectors (only u and v components)
    magnitude_2d = np.sqrt(u_slice**2 + v_slice**2)

    # Set up the 2D plot
    fig, ax = plt.subplots()

    # Normalize the magnitudes to [0, 1] for the colormap
    norm = plt.Normalize(magnitude_2d.min(), magnitude_2d.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(magnitude_2d))

    # Plot the 2D quiver plot using the u and v components
    ax.quiver(x, y, u_slice, v_slice, color=colors.reshape(-1, 4), angles='xy', scale_units='xy')

    # Add a colorbar to indicate the magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Magnitude')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Vector Field Slice at Z = 2.5 at index {z_slice}')

def plot_2d_scalar_slice(scalar_field):
    # Choose a specific Z slice (for example, the middle plane)
    z_slice = Z_SIZE // 2  # Taking the middle slice, but this can be any valid Z index

    sliced_values = scalar_field[:, :, z_slice]

    # Set up the 2D plot
    fig, ax = plt.subplots()

    cax = ax.imshow(sliced_values.T, extent=[0, 10, 0, 5], origin='lower')
    fig.colorbar(cax, ax=ax)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Scalar Field Slice at Z = 2.5 at index {z_slice}')


def parabolic_profile_2d(height, width, peak_velocity, num_points_height, num_points_width, edge_layers):
    # Generate y and z coordinates for the 2D profile
    y = np.linspace(-height/2, height/2, num_points_height)
    z = np.linspace(-width/2, width/2, num_points_width)
    Y, Z = np.meshgrid(y, z)

    # Compute the parabolic velocity profile
    velocity = peak_velocity * (1 - (Y**2 / (height/2)**2) - (Z**2 / (width/2)**2))
    
    # Ensure that velocity is non-negative (set negative values to zero)
    velocity = np.maximum(velocity, 0)

    # Apply zero velocity to the edge layers
    if edge_layers > 0:
        velocity[:edge_layers, :] = 0  # Top edge
        velocity[-edge_layers:, :] = 0  # Bottom edge
        velocity[:, :edge_layers] = 0  # Left edge
        velocity[:, -edge_layers:] = 0  # Right edge

    return velocity




    

grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [X_SIZE, Y_SIZE, Z_SIZE], periodic=[False, False, False])


# Define the mask for grid lines with thickness of 5 in 3D
x, y, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1], grid.cell_coords[..., 2]

y_count = 5
# z_count = 1
y_line_width_in_indices = 10  # TODO: Try to make this thicker
z_line_width_in_indices = 0

# Compute the size of each cell in indices
y_cell_size = Y_SIZE // y_count
# z_cell_size = Z_SIZE // z_count
y_indices = np.arange(Y_SIZE)
# z_indices = np.arange(Z_SIZE)

# Generate masks for y and z
y_mask = (y_indices % y_cell_size) < y_line_width_in_indices
# z_mask = (z_indices % z_cell_size) < z_line_width_in_indices
y_mask_3d = y_mask[np.newaxis, :, np.newaxis]
# z_mask_3d = z_mask[np.newaxis, np.newaxis, :]

# yz_mask = y_mask_3d | z_mask_3d
yz_mask = y_mask_3d
boundary_mask_yz = np.tile(yz_mask, (X_SIZE, 1, 1))
x_mask = (x >= 4.9) & (x <= 5.1)
boundary_mask = x_mask & boundary_mask_yz
# boundary_mask = x_mask & (y_mask[:, np.newaxis] | z_mask[np.newaxis, :])
boundary_mask = x_mask & y_mask[:, np.newaxis]

# Make outter edges of the grid (yz) be always true
boundary_mask[245:255+1, 0:y_line_width_in_indices+1, :] = False
boundary_mask[245:255+1, Y_SIZE-y_line_width_in_indices-1: Y_SIZE, :] = False
# boundary_mask[45:56+1, :, 0:z_line_width_in_indices+1] = False
# boundary_mask[45:55+1, :, Z_SIZE-z_line_widzth_in_indices-1:Z_SIZE] = False

plt.title("boundary mask x-y")
plt.imshow(boundary_mask[X_SIZE//2,:, :])
plt.show()

plt.title("boundary mask x-z")
plt.imshow(boundary_mask[:, Y_SIZE//2, :])
plt.show()

plt.title("boundary mask y-z")
plt.imshow(boundary_mask[:, :, Z_SIZE//2])
plt.show()


v0 = 1.0
v0_par = parabolic_profile_2d(5, 5, v0, Y_SIZE, Z_SIZE, 5)  # 5,5 needs to match the grid size
# print(v0_par)

init_density = 1*np.ones((X_SIZE, Y_SIZE, Z_SIZE))
# init_density = np.random.normal(loc=15, scale=0.01, size=(X_SIZE, Y_SIZE, Z_SIZE))

# init_density[:X_SIZE//2, :, :] = 5

scalar_field = pde.VectorField(grid, data=np.array([-v0,0.0,0.0])[:, np.newaxis, np.newaxis, np.newaxis])
# scalar_field = pde.VectorField(grid, data=0)
scalar_field.data[:, boundary_mask] = 0

# # Set the edge values at y and z boundaries to 0
# scalar_field.data[:, :, 0, :] = 0
# scalar_field.data[:, :, -1, :] = 0
# scalar_field.data[:, :, :, 0] = 0
# scalar_field.data[:, :, :, -1] = 0

density_field = pde.ScalarField(grid, data=init_density)
field = pde.FieldCollection([scalar_field, density_field])

# Set ALL x values to 1
# field.data[0, :, :, :] = 1

# bc_left_x = {"value": [v0, 0.0, 0.0]}       # Dirichlet condition on the left (x = 0)
# bc_right_x = {"derivative": 0}   # Neumann condition on the right (x = X_SIZE)
# bc_x_vec = [bc_left_x, bc_left_x] 
# bc_x_vec = ({"derivative": 0})
# bc_y = ({"derivative": 0})
# bc_z = ({"derivative": 0})

bc_x = ({"curvature": 0})
bc_y = ({"derivative": 0})
bc_z = ({"derivative": 0})

bc_left_density = {"value": 15}  # Dirichlet condition for density at x = 0
bc_right_density = {"derivative": 0} 
bc_x_density = [bc_left_density, bc_right_density]
bc_x_density = ({"curvature": 0})
bc_y_density = ({"derivative": 0})
bc_z_density = ({"derivative": 0})

eq = CustomPDE(bc=[bc_x, bc_y, bc_z], bc_vec=[bc_x, bc_y, bc_z], bc_density=[bc_x_density, bc_y_density, bc_z_density], boundary_mask=boundary_mask)

start_time = time.time()
storage = pde.MemoryStorage()
# result = eq.solve(field, t_range=20, dt=1e-2, explicit_fraction=0.5, solver="CrankNicolsonSolver")
result = eq.solve(field, t_range=500, dt=1e-2, adaptive=True, tracker=[
    storage.tracker(),
    pde.ProgressTracker(),
    LivePlotTracker2()
    ])
# result = eq.solve(field, t_range=1, dt=1e-2, adaptive=True)
end_time = time.time()
print("Execution Time: ", end_time - start_time, " seconds")

plot_2d_scalar_slice(result[1].data)
plot_2d_slice(result[0].data)

# Show the plot
plt.show()
# result[1].plot_interactive()
# result[0].to_scalar(scalar='norm').plot_interactive()