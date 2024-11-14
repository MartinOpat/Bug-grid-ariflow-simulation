import numpy as np
from matplotlib import pyplot as plt
import pde
from pde.tools.numba import jit
import time
import os

# Set bigger font of all matplotlib fonts
plt.rcParams.update({"font.size": 18})

def get_required_env_var(var_name):
    """Get the required environment variable or raise an error if not set."""
    try:
        return os.environ[var_name]
    except KeyError:
        raise EnvironmentError(f'The environment variable {var_name} is not set.')

LOG_NAME = get_required_env_var("LOG_NAME")
HOLE_COUNT = int(get_required_env_var("HOLE_COUNT"))
HOLE_WIDTH = float(get_required_env_var("HOLE_WIDTH"))

X_SIZE = 200 # should be divisible by 2
Y_SIZE = 100
Z_SIZE = 1

class CompressibleFlowPDE(pde.PDEBase):
    """
    Implementing the Navier-Stokes PDEs for compressible fluid flow
    to model the velocity field in a wind tunnel with a boundary_mask
    which blocks the fluid.

    Notes:
    ------
    Assuming the following physical assumptions:
    1. Mass conservation
    2. Isotropy (i.e., no gravity)
    3. Compressible fluid
    4. Newtonian fluid
    Additionally, using the mass continuity equation to model density
    in the fluid.
    """
    def __init__(self, boundary_mask):
        self.boundary_mask = np.ascontiguousarray(boundary_mask, dtype=np.bool_)
        self.dynamic_viscosity = 0.005
        self.bulk_viscosity = 0.1
        self.RT = 1
        self.artificial_viscosity_coefficient = 0.2

        self.laplace_u_op = None
        self.divergence_u_op = None
        self.gradient_u_op = None
        self.gradient_divergence_u_op = None

        self.check_implementation = False

        bc_x = "periodic"
        bc_y = {"value": 0}
        bc_z = {"value": 0}

        bc_x_density = "periodic"
        bc_y_density = {"derivative": 0}
        bc_z_density = {"derivative": 0}

        self.bc=[bc_x, bc_y, bc_z]
        self.bc_vec=[bc_x, bc_y, bc_z]
        self.bc_density=[bc_x_density, bc_y_density, bc_z_density]


    def evolution_rate(self, states, t=0):
        """
        This method computes the time derivatives of the velocity (u) and density (p) fields
        using the Navier-Stokes equations with added artificial viscosity and boundary conditions.

        Notes:
        ------
        - Artificial viscosity is added to stabilize the solution and prevent non-physical oscillations.
        - Boundary conditions are applied to ensure no-slip conditions at the boundaries.
        - Random noise is added to the velocity field at non-boundary cells to simulate turbulence.
        """
        u, p = states # velocity, density

        u.data[:, self.boundary_mask] = 0
        laplacian = u.laplace(bc=self.bc_vec)
        f_u = u.dot(u.gradient(bc=self.bc_vec)) - (self.dynamic_viscosity/p) * laplacian \
            - (self.bulk_viscosity/p + self.dynamic_viscosity / (3*p)) * u.divergence(bc=self.bc_vec).gradient(bc=self.bc)
        
        pressure = p*self.RT
        u_t = -f_u - pressure.gradient(bc=self.bc_density) / p.data[np.newaxis, :, :, :]

        artificial_viscosity = self.artificial_viscosity_coefficient * u.laplace(bc=self.bc_vec)
        u_t += artificial_viscosity
        u_t.data[0, ~self.boundary_mask] += np.random.normal(loc=0.1, scale=0.01, size=u_t.data[0, ~self.boundary_mask].shape)

        p_t = -(u*p).divergence(bc=self.bc_density)
        u_t.data[:, self.boundary_mask] = 0

        return pde.FieldCollection([u_t, p_t])

    def _make_pde_rhs_numba(self, states): 
        """
        This method creates and returns a JIT-compiled version of the evolution_rate method.
        """
        state_velocity, state_density = states
        if self.laplace_u_op is None:
            self.laplace_u_op = state_velocity.grid.make_operator("vector_laplace", bc=self.bc_vec)
        if self.divergence_u_op is None:
            self.divergence_u_op = state_velocity.grid.make_operator("divergence", bc=self.bc_vec)
        if self.gradient_u_op is None:
            self.gradient_u_op = state_velocity.grid.make_operator("vector_gradient", bc=self.bc_vec)
        if self.gradient_divergence_u_op is None:
            self.gradient_divergence_u_op = state_velocity.grid.make_operator("gradient", bc=self.bc)
        
        laplace_u_op = self.laplace_u_op
        divergence_u_op = self.divergence_u_op
        gradient_u_op = self.gradient_u_op
        gradient_divergence_u_op = self.gradient_divergence_u_op

        divergence_p_u_op = state_density.grid.make_operator("divergence", bc=self.bc_density)
        gradient_p_u = state_density.grid.make_operator("gradient", bc=self.bc_density)

        @jit(nopython=True)
        def convective_derivative(u, gradient_u_x, gradient_u_y, gradient_u_z):
            """
            Calculates the convective derivative D/Dt of a 3D vector field.

            @params:
            u: np.ndarray
                A 3D vector field.
            gradient_u_x: np.ndarray
                The x-component of the gradient of the vector field.
            gradient_u_y: np.ndarray
                The y-component of the gradient of the vector field.
            gradient_u_z: np.ndarray
                The z-component of the gradient of the vector field.

            """
            result = np.zeros_like(u)
            for i in range(3):
                result[i] = u[0] * gradient_u_x[i] + u[1] * gradient_u_y[i] + u[2] * gradient_u_z[i]
            return result


        dynamic_viscosity = self.dynamic_viscosity
        bulk_viscosity = self.bulk_viscosity
        boundary_mask = self.boundary_mask
        artificial_viscosity_coefficient = self.artificial_viscosity_coefficient
        RT = self.RT

        @jit(nopython=True)
        def apply_boundary_mask(vector_field, boundary_mask):
            """
            Apply a boundary mask to a 3D vector field.
            This function inplace sets the components of the vector field to zero at the positions
            where the boundary mask is True.

            @params:
            vector_field: np.ndarray
                A 3D vector field.
            boundary_mask: np.ndarray
                A 3D boolean mask.
            """
            for i in range(X_SIZE):
                for j in range(Y_SIZE):
                    for k in range(Z_SIZE):
                        if boundary_mask[i, j, k]:
                            vector_field[0, i, j, k] = 0
                            vector_field[1, i, j, k] = 0
                            vector_field[2, i, j, k] = 0

        
        @jit(nopython=True)
        def multiply_scalar_vector(scalar, vector_field):
            """
            Multiplies a scalar with a vector field element-wise.
            
            Returns a new matrix as the result.

            @params:
            scalar: float
                A scalar value.
            vector_field: np.ndarray
                A 3D vector field.
            """
            result = np.zeros_like(vector_field)
            for i in range(3):
                result[i] = scalar * vector_field[i]
            return result

        @jit(nopython=True)
        def multiply_scalar_vector_field(scalar_field, vector_field):
            """
            Multiplies each component of a 3D vector field by a corresponding scalar field.

            Returns a new matrix as the result.

            @params:
            scalar_field: np.ndarray
                A 3D scalar field.
            vector_field: np.ndarray
                A 3D vector field.
            """
            result = np.zeros_like(vector_field, dtype=np.float64)
            for x in range(X_SIZE):
                for y in range(Y_SIZE):
                    for z in range(Z_SIZE):
                        result[0][x][y][z] = vector_field[0][x][y][z]*scalar_field[x][y][z]
                        result[1][x][y][z] = vector_field[1][x][y][z]*scalar_field[x][y][z]
                        result[2][x][y][z] = vector_field[2][x][y][z]*scalar_field[x][y][z]

            return result

        @jit(nopython=True)
        def apply_force_to_x(vector_field, force):
            """
            Applies a force to the x-component of a 3D vector field.

            This function iterates over a 3D vector field and adds a specified force
            to the x-component of the vector at each point, except where the boundary
            mask is True.

            @params:
            vector_field: np.ndarray
                A 3D vector field.
            force: float
                The force to be applied to the x-component.
            """

            for x in range(X_SIZE):
                for y in range(Y_SIZE):
                    for z in range(Z_SIZE):
                        if not boundary_mask[x, y, z]:
                            vector_field[0, x, y, z] += force
            

        @jit(nopython=True)
        def pde_rhs(state_datas, t=0):
            """
            This method computes the time derivatives of the velocity (u) and density (p) fields
            using numba to run in a JIT-compiled way.

            Notes:
            ------
            Identical calculation as in the evolution_rate method.
            """
            
            u = state_datas[0:3] # velocity
            p = state_datas[3] # density

            apply_boundary_mask(u, boundary_mask)

            lapacian_u = laplace_u_op(u)
            gradient_u = gradient_u_op(u)
            gradient_divergence_u = gradient_divergence_u_op(divergence_u_op(u))
            p_u = multiply_scalar_vector_field(p, u)
            divergence_p_u = divergence_p_u_op(p_u)

            shear_viscosity = dynamic_viscosity/p
            bulk_kinematic_viscosity = bulk_viscosity/p

            con_dev = convective_derivative(u, gradient_u[0], gradient_u[1], gradient_u[2])

            shear_lap = multiply_scalar_vector(shear_viscosity, lapacian_u)

            vis_grad = multiply_scalar_vector(bulk_kinematic_viscosity + shear_viscosity / 3, gradient_divergence_u)

            f_u = con_dev - shear_lap - vis_grad

            pressure = p*RT
            u_t = -f_u - multiply_scalar_vector_field(1/p, gradient_p_u(pressure)) \
                  + lapacian_u*artificial_viscosity_coefficient

            force = np.random.normal(loc=0.1, scale=0.01)
            apply_force_to_x(u_t, force)

            apply_boundary_mask(u_t, boundary_mask)

            p_t = -divergence_p_u
            return np.concatenate((u_t, p_t[np.newaxis,:,:,:]), axis=0)

        return pde_rhs

class DensityLivePlot(pde.LivePlotTracker):
    """
    A tracker that displays a live updating plot of the density during calculation.
    """
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

class SpeedLivePlot(pde.LivePlotTracker):
    """
    A tracker that displays a plot of the magnitude of velocity during calculation.
    """
    grid_size = [X_SIZE, Y_SIZE]
    grid = pde.UnitGrid(grid_size)  
    z_slice = Z_SIZE // 2

    def initialize(self, state: pde.FieldBase, info = None) -> float:
        field_obj = pde.ScalarField(self.grid)
        return super().initialize(field_obj, info)

    def handle(self, state: pde.FieldBase, t: float) -> None:
        sliced_values = np.linalg.norm(state.data[:3], axis=0)[:, :, self.z_slice]
        field_obj = pde.ScalarField(self.grid, data=sliced_values)
        super().handle(field_obj, t)

def plot_2d_slice(vector_field):
    """
    Plots a 2D slice of a 3D vector field at the mid Z-plane in the form of a quiver plot.
    Notes:
    ------
    The function saves the plot as a PDF file in the 'results/velocity/' directory.

    @params:
    vector_field: np.ndarray
        A 3D vector field to be plotted.
    """
    
    # Extract the u, v, w components
    u = vector_field[0]
    v = vector_field[1]
    w = vector_field[2]

    # Choose a Z slice
    z_slice = Z_SIZE // 2

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
    """
    Plots a 2D slice of a 3D scalar field at the mid Z-plane in the form of a heatmap.
    Notes:
    ------
    The function saves the plot as a PDF file in the 'results/{name}/' directory.

    @params:
    scalar_field: np.ndarray
        A 3D scalar field to be plotted.
    name: str 
        The name of the field to be used in the directory name.
    """
    # Choose a Z slice
    z_slice = Z_SIZE // 2

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
dy = 5 / Y_SIZE # Y-direction Length of discretised cell

init_density = np.random.normal(loc=15, scale=0.01, size=(X_SIZE, Y_SIZE, Z_SIZE))

scalar_field = pde.VectorField(grid, data=0)
density_field = pde.ScalarField(grid, data=init_density)
field = pde.FieldCollection([scalar_field, density_field])

def get_bug_grid_mask():
    """
    Generates a mask for the bug grid based on the specified hole count and width.
    The function creates a mask for grid lines with a thickness of 0.2.
    In the middle of the x space i.e between 4.9 and 5.1 from a total range of 0 to 10.
    It calculates the mask based on the 
    number of holes (HOLE_COUNT) and the width of each hole (HOLE_WIDTH). The mask is used to determine the blocked 
    and empty spaces in the grid.
    Notes:
        - The function also calculates the percentage of the window blocked in the x-y slice of the window and prints it.
    """

    # Define the mask for grid lines with thickness of 5 in 3D
    x, y, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1], grid.cell_coords[..., 2]

    if HOLE_COUNT != 1:
        empty_space = HOLE_COUNT*HOLE_WIDTH

        assert 0 <= empty_space <= 5, "Invalid parameters for boundary mask. There is more hole than the total height of the tunnel."
        assert 1 <= HOLE_COUNT <= Y_SIZE//2-1, "Invalid parameters for boundary mask. There needs to be atleast one hole and no more than twice the resolution."
        assert HOLE_WIDTH >= 5/Y_SIZE, "Invalid parameters for boundary mask. The hole width must be atleast one grid cell wide."

        blocked_space = 5 - empty_space
        blocked_count = HOLE_COUNT - 1
        blocked_width = blocked_space/blocked_count
        y_mask = HOLE_WIDTH < y % (blocked_width + HOLE_WIDTH) 

        x_mask = (x >= 4.9) & (x <= 5.1)

        bug_grid_mask = y_mask & x_mask
    else:
        # Empty window -> boundary mask is false everywhere
        bug_grid_mask = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.bool_)


    # plt.title("boundary mask")
    # plt.imshow(bug_grid_mask[X_SIZE//2,:, :])
    # plt.show()
    # exit()

    # Calculate the % of window blocked in the x-y slice of the window
    window_blocked = np.sum(bug_grid_mask[X_SIZE//2,:, :]) / Y_SIZE
    print("Window blocked: ", window_blocked)
    return bug_grid_mask

bug_grid_idx = X_SIZE//2  # Index of the window tracker for the density
bug_grid_mask = get_bug_grid_mask()

def create_dm_dt_tracker(idx, dy, data_tracker_interval):
    """
    Create a data tracker for monitoring the rate of change of mass (dm/dt) during simulation.

    @params:
    idx: int
        The index of the bug grid mask.
    dy: float
        The length of the discretised cell in the y-direction.
    data_tracker_interval: float
        The interval at which the data tracker should record the rate of change of mass.
    """
    def get_dm_dt(states, time):
        state, state_density = states

        dm_dt = np.sum(state_density.data[idx] * state.data[0][idx] * dy)
        return {"dm_dt": dm_dt}

    dm_dt_tracker = pde.DataTracker(get_dm_dt, interval=data_tracker_interval)

    return dm_dt_tracker

eq = CompressibleFlowPDE(boundary_mask=bug_grid_mask)

storage = pde.MemoryStorage()

data_tracker_interval = 0.1
dm_dt_tracker = create_dm_dt_tracker(bug_grid_idx, dy, data_tracker_interval)

start_time = time.time()

result = eq.solve(field, t_range=60, dt=1e-2, scheme="euler", adaptive=True, tracker=[
    storage.tracker(),
    pde.ProgressTracker(),
    DensityLivePlot(),
    dm_dt_tracker,
    SpeedLivePlot(),
    ])
end_time = time.time()
print("Execution Time: ", end_time - start_time, " seconds")

# Save file whole history of data for further analysis
os.makedirs("results/raw_data", exist_ok=True)
np.save(f"results/raw_data/{LOG_NAME}.npy", storage.data)
np.save(f"results/raw_data/{LOG_NAME}_dm_dt.npy", dm_dt_tracker.data)

# Plot and save density and velocity fields
plot_2d_scalar_slice(result[1].data, "density")
plot_2d_scalar_slice(np.linalg.norm(result[0].data, axis=0), "velocity-magnitude")
plot_2d_slice(result[0].data)

def plot_dm_dt(dm_dt_tracker_data, data_tracker_interval):
    """
    Plots the rate of change of mass (dm/dt) over time and saves the plot as a PDF file.
    This function retrieves the dm/dt data from the dm_dt_tracker_data, calculates the corresponding
    time steps based on the data_tracker_interval, and generates a plot of dm/dt versus time.
    The plot is then saved in the "results/dm_dtOverTime" directory with the filename based on
    the LOG_NAME variable.

    @params:
    dm_dt_tracker_data: list
        A list of dictionaries containing the dm/dt data.
    data_tracker_interval: float
        The interval at which the data tracker recorded the rate of change of mass
    """

    dm_dts = []
    for dm_dt in dm_dt_tracker_data:
        dm_dts.append(dm_dt['dm_dt'])
    dm_dts = np.array(dm_dts)
    ts = np.arange(start=0, stop=len(dm_dts)*data_tracker_interval, step=data_tracker_interval)

    plt.figure()
    plt.title("Q vs time")
    plt.xlabel("Time")
    plt.ylabel("Q")
    plt.plot(ts, dm_dts)
    os.makedirs("results/dm_dtOverTime", exist_ok=True)
    plt.savefig(f"results/dm_dtOverTime/{LOG_NAME}.pdf")

plot_dm_dt(dm_dt_tracker.data, data_tracker_interval)

def get_z_slice_density_movie(storage, z_slice=Z_SIZE // 2):
    """
    Generates a storage of slices along the z-axis from the given storage data.

    @params:
    storage: pde.storage.memory.MemoryStorage
        The storage object containing the simulation data.
    z_slice: int
        The index of the z-slice to extract from the 3D data.
    """
    new_data = []
    for time in range(len(storage)):
        sliced_values = storage[time].data[3][:, :, z_slice]
        new_data.append(sliced_values)
    new_data = np.array(new_data)
    grid_size = [X_SIZE, Y_SIZE]
    grid = pde.UnitGrid(grid_size)  

    field_obj = pde.ScalarField(grid, data=new_data[0])
    res = pde.storage.memory.MemoryStorage(times=list(range(len(storage))), data=new_data, field_obj=field_obj)

    return res

new_storage2 = get_z_slice_density_movie(storage)

# Export movie of density data evolving throughout the simulation.
vmin = np.min(new_storage2.data)
vmax = np.max(new_storage2.data)

os.makedirs("results/movie", exist_ok=True)
pde.movie(new_storage2, filename=f"results/movie/{LOG_NAME}.mp4", plot_args={"vmin": vmin, "vmax": vmax}, movie_args={})
