import numpy as np
from matplotlib import pyplot as plt
import pde
from pde.tools.numba import jit
from numba import prange
import time
from scipy.ndimage import gaussian_filter, convolve, uniform_filter

X_SIZE = 100 # should be divisible by 2
Y_SIZE = 50
Z_SIZE = 50

class CustomPDE(pde.PDEBase):

    def __init__(self, bc, bc_density, boundary_mask, bc_vec, fan):
        self.bc = bc
        self.boundary_mask = np.ascontiguousarray(boundary_mask, dtype=np.bool_)
        self.vector_array = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
        self.vector_array[0, -1 + X_SIZE//2:1 + X_SIZE//2, :, :] = 10
        self.vector_array = np.ascontiguousarray(self.vector_array, dtype=np.float64)
        self.fan = fan

        self.dynamic_viscosity = 0.01
        self.bulk_viscosity = 1
        self.bc_vec = bc_vec
        self.RT = 1

        self.bc_density = bc_density

        self.laplace_u = None
        self.divergence_u = None
        self.gradient_u = None
        self.gradient_u_2 = None

    def evolution_rate(self, states, t=0):
        state, density = states
        density.data = uniform_filter(density.data, size=4)
        laplacian = state.laplace(bc=self.bc_vec)
        f_u = state.dot(state.gradient(bc=self.bc_vec)) - (self.dynamic_viscosity/density) * laplacian \
              - (self.bulk_viscosity/density + self.dynamic_viscosity / (3*density)) * state.divergence(bc=self.bc_vec).gradient(bc=self.bc)
        
        pressure = density*self.RT
        ans = -f_u - pressure.gradient(bc=self.bc_density) / density.data[np.newaxis, :, :, :]
        ans.data[:, self.boundary_mask] = 0
        ans.data[0, self.fan] += 0.1
        density_derivative = -(state*density).divergence(bc=self.bc_density)

        return pde.FieldCollection([ans, density_derivative])

    # def _make_pde_rhs_numba(self, states): 
        state, state_density = states
        if self.laplace_u is None:
            self.laplace_u = state.grid.make_operator("vector_laplace", bc=self.bc_vec)
        if self.divergence_u is None:
            self.divergence_u = state.grid.make_operator("divergence", bc=self.bc_vec)
        if self.gradient_u is None:
            self.gradient_u = state.grid.make_operator("vector_gradient", bc=self.bc_vec)
        if self.gradient_u_2 is None:
            self.gradient_u_2 = state.grid.make_operator("gradient", bc=self.bc)
        
        laplace_u = self.laplace_u
        divergence_u = self.divergence_u
        gradient_u = self.gradient_u
        gradient_u_2 = self.gradient_u_2

        divergence_p_u = state_density.grid.make_operator("divergence", bc=self.bc_density)
        gradient_p_u = state_density.grid.make_operator("gradient", bc=self.bc_density)

        @jit(nopython=True, parallel=True)
        def convective_derivative(u, gradient_u_x, gradient_u_y, gradient_u_z):
            result = np.zeros_like(u)
            for i in prange(3):
                result[i] = u[0] * gradient_u_x[i] + u[1] * gradient_u_y[i] + u[2] * gradient_u_z[i]
            return result


        dynamic_viscosity = self.dynamic_viscosity
        bulk_viscosity = self.bulk_viscosity
        vector_array = self.vector_array
        boundary_mask = self.boundary_mask
        fan = self.fan
        # apply_boundary_mask = self.apply_boundary_mask
        RT = self.RT

        @jit(nopython=True, parallel=True)
        def apply_boundary_mask(vector_field, boundary_mask):
            for i in prange(X_SIZE):
                for j in range(Y_SIZE):
                    for k in range(Z_SIZE):
                        if boundary_mask[i, j, k]:
                            vector_field[0, i, j, k] = 0
                            vector_field[1, i, j, k] = 0
                            vector_field[2, i, j, k] = 0

        @jit(nopython=True, parallel=True)
        def apply_fan(vector_field, fan):
            for i in prange(X_SIZE):
                for j in range(Y_SIZE):
                    for k in range(Z_SIZE):
                        if fan[i, j, k]:
                            vector_field[0, i, j, k] += 0.1
        
        @jit(nopython=True, parallel=True)
        def multiply_scalar_vector(scalar, vector):
            result = np.zeros_like(vector)
            for i in range(3):
                result[i] = scalar * vector[i]
            return result

        @jit(nopython=True, parallel=True)
        def multiply_scalar_vector_field(scalar_field, vector_field):
            result = np.zeros_like(vector_field, dtype=np.float64)
            for x in prange(X_SIZE):
                for y in range(Y_SIZE):
                    for z in range(Z_SIZE):
                        result[0][x][y][z] = vector_field[0][x][y][z]*scalar_field[x][y][z]
                        result[1][x][y][z] = vector_field[1][x][y][z]*scalar_field[x][y][z]
                        result[2][x][y][z] = vector_field[2][x][y][z]*scalar_field[x][y][z]

            return result


        @jit(nopython=True, parallel=True)
        def pde_rhs(state_datas, t=0):
            state_data = state_datas[0:3]
            state_density_data = state_datas[3]
            # print(type(state_density_data))
            # state_data, state_density_data = state_datas
            state_lapacian = laplace_u(state_data)
            state_grad = gradient_u(state_data)
            state_grad2 = gradient_u_2(divergence_u(state_data))
            product_p_u = multiply_scalar_vector_field(state_density_data, state_data)
            state_divergence_p = divergence_p_u(product_p_u)

            shear_viscosity = dynamic_viscosity/state_density_data
            bulk_kinematic_viscosity = bulk_viscosity/state_density_data

            con_dev = convective_derivative(state_data, state_grad[0], state_grad[1], state_grad[2])

            shear_lap = multiply_scalar_vector(shear_viscosity, state_lapacian)

            vis_grad2 = multiply_scalar_vector(bulk_kinematic_viscosity + shear_viscosity / 3, state_grad2)

            f_u = con_dev - shear_lap - vis_grad2

            pressure = state_density_data*RT
            # ans = -f_u - pressure.gradient(bc=self.bc_density) / density.data[np.newaxis, :, :, :]
            ans = -f_u - multiply_scalar_vector_field(1/state_density_data, gradient_p_u(pressure))

            apply_boundary_mask(ans, boundary_mask)

            density_t = -state_divergence_p
            # return np.stack([ans, density_t])
            apply_fan(ans, fan)
            return np.concatenate((ans, density_t[np.newaxis,:,:,:]), axis=0)

        return pde_rhs

class LivePlotTracker2(pde.LivePlotTracker):
    grid_size = [X_SIZE, Y_SIZE]
    grid = pde.UnitGrid(grid_size)  
    z_slice = Z_SIZE // 2

    def initialize(self, state: pde.FieldBase, info = None) -> float:
        field_obj = pde.ScalarField(self.grid)
        return super().initialize(field_obj, info)

    def handle(self, state: pde.FieldBase, t: float) -> None:
        sliced_values = state.data[3][:, :, self.z_slice]
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

grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [X_SIZE, Y_SIZE, Z_SIZE], periodic=[False, False, False])
init_density = 15*np.ones((X_SIZE, Y_SIZE, Z_SIZE))
# init_density = np.random.normal(loc=15, scale=0.01, size=(X_SIZE, Y_SIZE, Z_SIZE))

# init_density[:X_SIZE//2, :, :] = 5
scalar_field = pde.VectorField(grid, data=0)
density_field = pde.ScalarField(grid, data=init_density)
field = pde.FieldCollection([scalar_field, density_field])

# Set ALL x values to 1
# field.data[0, :, :, :] = 1

# bc_left_x = {"value": [0.1, 0, 0]}       # Dirichlet condition on the left (x = 0)
# bc_right_x = {"derivative": 0}   # Neumann condition on the right (x = X_SIZE)
# bc_x_vec = [bc_left_x, bc_right_x] 
bc_x_vec = {"derivative": 0}   # Neumann condition on the right (x = X_SIZE)
bc_y = ( {"derivative": 0})
bc_z = ( {"derivative": 0})

bc_x = {"derivative": 0}
bc_y = ( {"derivative": 0})
bc_z = ( {"derivative": 0})

bc_left_density = {"value": 15}  # Dirichlet condition for density at x = 0
bc_right_density = {"derivative": 0} 
bc_x_density = [bc_left_density, bc_right_density]
bc_x_density = ( {"derivative": 0})
bc_y_density = ( {"derivative": 0})
bc_z_density = ( {"derivative": 0})

# Define the mask for grid lines with thickness of 5 in 3D
x, y, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1], grid.cell_coords[..., 2]
# print(x.shape)
# section_size = 10 // 3
# print((4 <= x) & (x <= 6))

y_width = 0.5
y_count = 5
z_width = 0.5
z_count = 5

# assert(10/Y_SIZE < y_count and 10/Z_SIZE < z_count)

y_empty_width = 5/y_count - y_width
y_total_width = y_empty_width + y_count
z_empty_width = 5/z_count - z_width
z_total_width = z_empty_width + z_count

boundary_mask = (
    ((4 <= x) & (x <= 6)) &
    # (((y % section_size <= 2) | (y % section_size >= section_size - 2)) |
    # (((2 <= y) & (y <= 3)) |
    (((y+y_total_width/2)%y_total_width <= y_width) |
    # ((y%(5/y_count) >= 5/y_count-y_width/2) | (y%(5/y_count) <= y_width/2) |
    # ((z % section_size <= 2) | (z % section_size >= section_size - 2)))
    # ((2 <= z) & (z <= 3)))
    ((z+z_total_width/2)%z_total_width <= z_width))
    # (z%(5/z_count) >= 5/z_count-z_width/2) | (z%(5/z_count) <= z_width/2))
)
# boundary_mask = np.zeros((X_SIZE, Y_SIZE, Z_SIZE))

fan = (
    (1.5 <= x) & (x <= 2.5) &
    (2.75 <= y) & (y <= 3.25) &
    (2.75 <= z) & (z <= 4.5)
)


plt.title("boundary mask")
plt.imshow(boundary_mask[X_SIZE//2,:, :])
plt.show()

eq = CustomPDE(
    bc=[bc_y, bc_x, bc_z],
    bc_vec=[bc_y, bc_x_vec, bc_z],
    bc_density=[bc_x_density, bc_y_density, bc_z_density],
    boundary_mask=boundary_mask,
    fan=fan
    )

start_time = time.time()
storage = pde.MemoryStorage()
# result = eq.solve(field, t_range=20, dt=1e-2, explicit_fraction=0.5, solver="CrankNicolsonSolver")
result = eq.solve(field, t_range=100, dt=1e-2, adaptive=True, tracker=[
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

# Export the movie
# Cross-section
def get_slice(storage, slice=32):
    # Transform the 3D storage data to 2D slice at depth z
    grid_size = [100, 100]
    grid = pde.UnitGrid(grid_size)  
    
    new_data = []
    for time in range(len(storage)):
        # data=storage[time].data[:,:,slice]
        norm = np.linalg.norm(storage[time].data, axis=0)
        data=norm[slice, :, :]

        new_data.append(data)
    new_data = np.array(new_data)
    field_obj = pde.ScalarField(grid, data=new_data[0])
    res = pde.storage.memory.MemoryStorage(times=list(range(len(storage))), data=new_data, field_obj=field_obj)
    # print("res",res[0])
        
    return res

def get_z_slice_density_movie(storage, z_slice = Z_SIZE // 2):
    new_data = []

    grid_size = [X_SIZE, Y_SIZE]
    grid = pde.UnitGrid(grid_size)  
    for time in range(len(storage)):
        # data=storage[time].data[:,:,slice]

        sliced_values = storage[time].data[3][:, :, z_slice]
        new_data.append(sliced_values)

    new_data = np.array(new_data)
    field_obj = pde.ScalarField(grid, data=new_data[0])
    res = pde.storage.memory.MemoryStorage(times=list(range(len(storage))), data=new_data, field_obj=field_obj)

    return res

# new_storage = get_slice(storage, slice=50)
# pde.movie(new_storage, filename="output3.mp4", plot_args={}, movie_args={})
new_storage2 = get_z_slice_density_movie(storage)

vmin = np.min(new_storage2.data)
vmax = np.max(new_storage2.data)

pde.movie(new_storage2, filename="densityzslice.mp4", plot_args={"vmin": vmin, "vmax": vmax}, movie_args={})