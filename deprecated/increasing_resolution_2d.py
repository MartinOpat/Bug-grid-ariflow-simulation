import numpy as np
from matplotlib import pyplot as plt
import pde

X_SIZE = 80
Y_SIZE = 40
Z_SIZE = 40

class CustomPDE(pde.PDEBase):

    def __init__(self, bc, boundary_mask):
        array_shape = (X_SIZE, Y_SIZE, Z_SIZE)
        # self.x_component_array = np.arange(array_shape[0]).reshape(array_shape[0], 1, 1) * np.ones(array_shape[1:])
        self.bc = bc
        self.boundary_mask = boundary_mask
        # self.boundary_mask = boundary_mask

        # array_shape = (20, 10, 10)
        # x_component_array = np.arange(array_shape[0]).reshape(array_shape[0], 1, 1) * np.ones(array_shape[1:])

        # # Create a new array for the vectors
        # self.vector_array = np.zeros((3, 20, 10, 10))

        # # Fill in the x-component and set y and z components to zero
        # self.vector_array[0] = x_component_array  # x component
        # y and z components remain zero by default
        self.vector_array = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE))
        self.vector_array[0] = 1

    def evolution_rate(self, state, t=0):
        """ Custom PDE evolution with modified diffusion rate """
        # laplacian = state.laplace(bc={"value": 0})  # zero Dirichlet boundary condition
        # rate = laplacian.copy()  # Default rate inside the grid
        
        # Reduce the rate of diffusion at grid to a quarter
        # rate.data[self.boundary_mask] *= 0.25
        # shear_viscosity = 2E-5
        # bulk_viscosity = 5E-5
        shear_viscosity = 0.1
        bulk_viscosity = 0.1
        f_u = state.dot(state.gradient(bc=self.bc)) - shear_viscosity * state.laplace(bc=self.bc) \
                - (bulk_viscosity + shear_viscosity/3) * state.divergence(bc=self.bc).gradient(bc=self.bc) 
        
        ans = -f_u - 0.1 * self.vector_array
        ans.data[:, self.boundary_mask] = 0
        return ans


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
    x, y = np.meshgrid(np.arange(X_SIZE), np.arange(Y_SIZE), indexing='ij')

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
    ax.set_title(f'2D Vector Field Slice at Z = {z_slice}')

grid = pde.CartesianGrid([[0, 10], [0, 5], [0, 5]], [X_SIZE, Y_SIZE, Z_SIZE], periodic=[False, False, False])
field = pde.VectorField(grid, data=0)
# plot_vector_field(field.data)
bc_x = ( {"value": 0})


# Define the mask for grid lines with thickness of 5 in 3D
x, y, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1], grid.cell_coords[..., 2]
# print(x.shape)
# section_size = 10 // 3
# print((4 <= x) & (x <= 6))
y_width = 0.5
y_count = 5
z_width = 0.5
z_count = 5
boundary_mask = (
    ((4 <= x) & (x <= 6)) &
    # (((y % section_size <= 2) | (y % section_size >= section_size - 2)) |
    # (((2 <= y) & (y <= 3)) |
    # ((y%(y_width + y_count) <= y_width) |
    ((y%(5/y_count) >= 5/y_count-y_width/2) | (y%(5/y_count) <= y_width/2) |
    # ((z % section_size <= 2) | (z % section_size >= section_size - 2)))
    # ((2 <= z) & (z <= 3)))
    # (z%(z_width + z_count) <= z_width))
    (z%(5/z_count) >= 5/z_count-z_width/2) | (z%(5/z_count) <= z_width/2))
)
plt.title("boundary mask")
plt.imshow(boundary_mask[X_SIZE//2,:, :])

eq = CustomPDE(bc=[bc_x, bc_x, bc_x], boundary_mask=boundary_mask)
result = eq.solve(field, t_range=1, dt=0.001)

plot_2d_slice(result.data)

# result.to_scalar(scalar='norm').plot_interactive()

# Show the plot
plt.show()