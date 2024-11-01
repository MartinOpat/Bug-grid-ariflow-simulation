import numpy as np
from matplotlib import pyplot as plt
import pde
from pde.tools.numba import jit
from numba import prange
from scipy.ndimage import gaussian_filter, convolve, uniform_filter
from scipy.integrate import trapezoid
import time
import os

def get_required_env_var(var_name):
    """Get the required environment variable or raise an error if not set."""
    try:
        return os.environ[var_name]
    except KeyError:
        raise EnvironmentError(f'The environment variable {var_name} is not set.')

LOG_NAME = get_required_env_var("LOG_NAME")

X_SIZE = 200 # should be divisible by 2
Y_SIZE = 100
Z_SIZE = 1

class CustomPDE(pde.PDEBase):

    def __init__(self, bc, bc_density, boundary_mask, bc_vec):
        self.bc = bc
        self.boundary_mask = np.ascontiguousarray(boundary_mask, dtype=np.bool_)
        self.vector_array = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
        self.vector_array[0, -1 + X_SIZE//2:1 + X_SIZE//2, :, :] = 10
        self.vector_array = np.ascontiguousarray(self.vector_array, dtype=np.float64)
        self.dynamic_viscosity = 0.005
        self.bulk_viscosity = 0.1
        self.bc_vec = bc_vec
        self.RT = 1
        self.artificial_viscosity_coefficient = 0.2
    
        self.bc_density = bc_density

        self.laplace_u = None
        self.divergence_u = None
        self.gradient_u = None
        self.gradient_u_2 = None

        self.check_implementation = False

    def evolution_rate(self, states, t=0):
        state, density = states

        state.data[:, self.boundary_mask] = 0
                # Cap speed at \sqrt{3} v0
        # state.data[0, :, :, :] = np.clip(state.data[0, :, :, :], -10, 10)
        # state.data[1, :, :, :] = np.clip(state.data[1, :, :, :], -10, 10)
        # state.data[2, :, :, :] = np.clip(state.data[2, :, :, :], -10, 10)

        laplacian = state.laplace(bc=self.bc_vec)
        f_u = state.dot(state.gradient(bc=self.bc_vec)) - (self.dynamic_viscosity/density) * laplacian \
            - (self.bulk_viscosity/density + self.dynamic_viscosity / (3*density)) * state.divergence(bc=self.bc_vec).gradient(bc=self.bc)
        
        pressure = density*self.RT
        ans = -f_u - pressure.gradient(bc=self.bc_density) / density.data[np.newaxis, :, :, :]

        artificial_viscosity = self.artificial_viscosity_coefficient * state.laplace(bc=self.bc_vec)
        ans += artificial_viscosity
        # ans.data[0, ~self.boundary_mask] += 0.1
        ans.data[0, ~self.boundary_mask] += np.random.normal(loc=0.1, scale=0.01, size=ans.data[0, ~self.boundary_mask].shape)


        # ans.data[:, self.boundary_mask] = 0
        density_derivative = -(state*density).divergence(bc=self.bc_density)
        # density_derivative.data[self.boundary_mask] = 0
        ans.data[:, self.boundary_mask] = 0


        return pde.FieldCollection([ans, density_derivative])

    def _make_pde_rhs_numba(self, states): 
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
        artificial_viscosity_coefficient = self.artificial_viscosity_coefficient
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
        def apply_force_to_x(vector_field, force):
            for x in prange(X_SIZE):
                for y in range(Y_SIZE):
                    for z in range(Z_SIZE):
                        if not boundary_mask[x, y, z]:
                            vector_field[0, x, y, z] += force
            

        @jit(nopython=True, parallel=True)
        def pde_rhs(state_datas, t=0):
            state_data = state_datas[0:3]
            state_density_data = state_datas[3]
            apply_boundary_mask(state_data, boundary_mask)
            # state_data = clamp_vector_field(state_data, 10)
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

            ans += state_lapacian*artificial_viscosity_coefficient


            # apply_force_to_x(ans, 0.1)
            force = np.random.normal(loc=0.1, scale=0.01)
            apply_force_to_x(ans, force)

            apply_boundary_mask(ans, boundary_mask)

            density_t = -state_divergence_p
            # return np.stack([ans, density_t])
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

class LivePlotTrackerVf(pde.LivePlotTracker):
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
    fig.set_size_inches(18.5, 10.5)

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
    os.makedirs("results/velocity/", exist_ok=True)
    fig.savefig(f"results/velocity/{LOG_NAME}.pdf", dpi=100)

def plot_2d_scalar_slice(scalar_field, name):
    # Choose a specific Z slice (for example, the middle plane)
    z_slice = Z_SIZE // 2  # Taking the middle slice, but this can be any valid Z index

    sliced_values = scalar_field[:, :, z_slice]

    # Set up the 2D plot
    fig, ax = plt.subplots()

    cax = ax.imshow(sliced_values.T, extent=[0, 10, 0, 5], origin='lower')
    fig.colorbar(cax, ax=ax)
    fig.set_size_inches(18.5, 10.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Scalar Field Slice at Z = 2.5 at index {z_slice}')
    os.makedirs(f"results/{name}", exist_ok=True)
    fig.savefig(f"results/{name}/{LOG_NAME}.pdf", dpi=100)

grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [X_SIZE, Y_SIZE, Z_SIZE], periodic=[True, False, False])
dy = 5 / Y_SIZE
# init_density = 15*np.ones((X_SIZE, Y_SIZE, Z_SIZE))
init_density = np.random.normal(loc=15, scale=0.01, size=(X_SIZE, Y_SIZE, Z_SIZE))

# init_density[:X_SIZE//2, :, :] = 5
scalar_field = pde.VectorField(grid, data=0)
density_field = pde.ScalarField(grid, data=init_density)
field = pde.FieldCollection([scalar_field, density_field])

# Set ALL x values to 1
# field.data[0, :, :, :] = 1

bc_left_x = {"value": [0.1, 0, 0]}       # Dirichlet condition on the left (x = 0)
bc_right_x = {"derivative": 0}   # Neumann condition on the right (x = X_SIZE)
bc_x_vec = [bc_left_x, bc_right_x] 
bc_y = ( {"derivative": 0})
bc_z = ( {"derivative": 0})

bc_x = "periodic"
bc_y = ( {"value": 0})
bc_z = ( {"value": 0})

bc_left_density = {"value": 15}  # Dirichlet condition for density at x = 0
bc_right_density = {"derivative": 0} 
bc_x_density = [bc_left_density, bc_right_density]
bc_x_density = "periodic"
bc_y_density = ( {"derivative": 0})
bc_z_density = ( {"derivative": 0})




# Define the mask for grid lines with thickness of 5 in 3D
x, y, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1], grid.cell_coords[..., 2]

hole_count = int(get_required_env_var("HOLE_COUNT"))
hole_width = float(get_required_env_var("HOLE_WIDTH"))

if hole_count != 1:
    empty_space = hole_count*hole_width

    assert 0 <= empty_space <= 5, "Invalid parameters for boundary mask. There is more hole than the total height of the tunnel."
    assert 1 <= hole_count <= Y_SIZE//2-1, "Invalid parameters for boundary mask. There needs to be atleast one hole and no more than twice the resolution."
    assert hole_width >= 5/Y_SIZE, "Invalid parameters for boundary mask. The hole width must be atleast one grid cell wide."

    blocked_space = 5 - empty_space
    blocked_count = hole_count - 1
    blocked_width = blocked_space/blocked_count
    y_mask = hole_width < y % (blocked_width + hole_width) 

    x_mask = (x >= 4.9) & (x <= 5.1)

    boundary_mask = y_mask & x_mask
else:
    # Empty window -> boundary mask is false everywhere
    boundary_mask = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.bool_)

idx = X_SIZE//2  # Index of the window tracker for the density

plt.title("boundary mask")
plt.imshow(boundary_mask[X_SIZE//2,:, :])
plt.show()
exit()

eq = CustomPDE(bc=[bc_x, bc_y, bc_z], bc_vec=[bc_x, bc_y, bc_z], bc_density=[bc_x_density, bc_y_density, bc_z_density], boundary_mask=boundary_mask)

start_time = time.time()
storage = pde.MemoryStorage()
def get_statistics(states, time):
    global idx, dy
    state, state_density = states

    dm_dt = np.sum(state_density.data[idx] * state.data[0][idx] * dy)
    # print("dm_dt", dm_dt)
    return {"dm_dt": dm_dt}

data_tracker_interval = 0.1
data_tracker = pde.DataTracker(get_statistics, interval=data_tracker_interval)

result = eq.solve(field, t_range=60, dt=1e-2, adaptive=True, tracker=[
    storage.tracker(),
    pde.ProgressTracker(),
    # LivePlotTracker2(),
    data_tracker,
    # LivePlotTrackerVf(),
    ])
# result = eq.solve(field, t_range=1, dt=1e-2, adaptive=True)
end_time = time.time()
print("Execution Time: ", end_time - start_time, " seconds")

os.makedirs("results/raw_data", exist_ok=True)
np.save(f"results/raw_data/{LOG_NAME}.npy", storage.data)

plot_2d_scalar_slice(result[1].data, "density")
plot_2d_scalar_slice(np.linalg.norm(result[0].data, axis=0), "velocity-magnitude")
plot_2d_slice(result[0].data)

dm_dts = []
for dm_dt in data_tracker.data:
    dm_dts.append(dm_dt['dm_dt'])

dm_dts = np.array(dm_dts)
ts = np.arange(start=0, stop=len(dm_dts)*data_tracker_interval, step=data_tracker_interval)

plt.figure()
plt.title("dm_dt vs time")
plt.plot(ts, dm_dts)
os.makedirs("results/dm_dtOverTime", exist_ok=True)
plt.savefig(f"results/dm_dtOverTime/{LOG_NAME}.pdf")

# plt.show()
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

os.makedirs("results/movie", exist_ok=True)
pde.movie(new_storage2, filename=f"results/movie/{LOG_NAME}.mp4", plot_args={"vmin": vmin, "vmax": vmax}, movie_args={})