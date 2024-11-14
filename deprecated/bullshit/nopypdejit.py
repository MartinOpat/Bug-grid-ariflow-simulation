import numpy as np
import time
from numba import njit, prange

# Grid parameters
X_SIZE = 160  # Should be divisible by 2
Y_SIZE = 80
Z_SIZE = 80
X_RANGE = (0, 10)
Y_RANGE = (0, 5)
Z_RANGE = (0, 5)

dx = (X_RANGE[1] - X_RANGE[0]) / X_SIZE
dy = (Y_RANGE[1] - Y_RANGE[0]) / Y_SIZE
dz = (Z_RANGE[1] - Z_RANGE[0]) / Z_SIZE

x = np.linspace(X_RANGE[0] + dx/2, X_RANGE[1] - dx/2, X_SIZE)
y = np.linspace(Y_RANGE[0] + dy/2, Y_RANGE[1] - dy/2, Y_SIZE)
z = np.linspace(Z_RANGE[0] + dz/2, Z_RANGE[1] - dz/2, Z_SIZE)

x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')

# Initial field u
u = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)

# Initialize vector_array
vector_array = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
center_slice = slice(-1 + X_SIZE//2, 1 + X_SIZE//2)
vector_array[0, center_slice, :, :] = 1  # Set x-component to 1 at the center

# Boundary mask
y_width = 0.5
y_count = 5
z_width = 0.5
z_count = 5
boundary_mask = (
    ((3 <= x_grid) & (x_grid <= 7)) &
    (
        ((y_grid % (5 / y_count) >= 5 / y_count - y_width / 2) | (y_grid % (5 / y_count) <= y_width / 2)) |
        ((z_grid % (5 / z_count) >= 5 / z_count - z_width / 2) | (z_grid % (5 / z_count) <= z_width / 2))
    )
)
boundary_mask = boundary_mask.astype(np.bool_)

# Shear and bulk viscosity
shear_viscosity = 0.1
bulk_viscosity = 0.1

# Time parameters
t_range = 0.5
dt = 0.01
num_steps = int(t_range / dt)

# Precompute deltas
deltas = np.array([dx, dy, dz])

# Numba-optimized helper functions
@njit(parallel=True)
def derivative(f, axis, delta):
    df = np.zeros_like(f)
    nx, ny, nz = f.shape
    if axis == 0:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 0 < i < nx - 1:
                        df[i, j, k] = (f[i + 1, j, k] - f[i - 1, j, k]) / (2 * delta)
                    elif i == 0:
                        df[i, j, k] = (f[i + 1, j, k] - f[i, j, k]) / delta
                    else:
                        df[i, j, k] = (f[i, j, k] - f[i - 1, j, k]) / delta
    elif axis == 1:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 0 < j < ny - 1:
                        df[i, j, k] = (f[i, j + 1, k] - f[i, j - 1, k]) / (2 * delta)
                    elif j == 0:
                        df[i, j, k] = (f[i, j + 1, k] - f[i, j, k]) / delta
                    else:
                        df[i, j, k] = (f[i, j, k] - f[i, j - 1, k]) / delta
    else:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 0 < k < nz - 1:
                        df[i, j, k] = (f[i, j, k + 1] - f[i, j, k - 1]) / (2 * delta)
                    elif k == 0:
                        df[i, j, k] = (f[i, j, k + 1] - f[i, j, k]) / delta
                    else:
                        df[i, j, k] = (f[i, j, k] - f[i, j, k - 1]) / delta
    return df

@njit(parallel=True)
def second_derivative(f, axis, delta):
    d2f = np.zeros_like(f)
    nx, ny, nz = f.shape
    if axis == 0:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 1 <= i <= nx - 2:
                        d2f[i, j, k] = (f[i + 1, j, k] - 2 * f[i, j, k] + f[i - 1, j, k]) / (delta ** 2)
                    elif i == 0:
                        d2f[i, j, k] = (f[i + 1, j, k] - 2 * f[i, j, k]) / (delta ** 2)
                    else:
                        d2f[i, j, k] = (-2 * f[i, j, k] + f[i - 1, j, k]) / (delta ** 2)
    elif axis == 1:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 1 <= j <= ny - 2:
                        d2f[i, j, k] = (f[i, j + 1, k] - 2 * f[i, j, k] + f[i, j - 1, k]) / (delta ** 2)
                    elif j == 0:
                        d2f[i, j, k] = (f[i, j + 1, k] - 2 * f[i, j, k]) / (delta ** 2)
                    else:
                        d2f[i, j, k] = (-2 * f[i, j, k] + f[i, j - 1, k]) / (delta ** 2)
    else:
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if 1 <= k <= nz - 2:
                        d2f[i, j, k] = (f[i, j, k + 1] - 2 * f[i, j, k] + f[i, j, k - 1]) / (delta ** 2)
                    elif k == 0:
                        d2f[i, j, k] = (f[i, j, k + 1] - 2 * f[i, j, k]) / (delta ** 2)
                    else:
                        d2f[i, j, k] = (-2 * f[i, j, k] + f[i, j, k - 1]) / (delta ** 2)
    return d2f

@njit(parallel=True)
def compute_gradient(u, deltas):
    gradient_u = np.zeros((3, 3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for comp in prange(3):
        for axis in range(3):
            gradient_u[comp, axis] = derivative(u[comp], axis, deltas[axis])
    return gradient_u

@njit(parallel=True)
def compute_divergence(u, deltas):
    divergence_u = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for axis in prange(3):
        divergence_u += derivative(u[axis], axis, deltas[axis])
    return divergence_u

@njit(parallel=True)
def compute_gradient_scalar_field(f, deltas):
    grad_f = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for axis in prange(3):
        grad_f[axis] = derivative(f, axis, deltas[axis])
    return grad_f

@njit(parallel=True)
def compute_laplacian(u, deltas):
    laplacian_u = np.zeros_like(u)
    for comp in prange(3):
        for axis in range(3):
            laplacian_u[comp] += second_derivative(u[comp], axis, deltas[axis])
    return laplacian_u

@njit(parallel=True)
def evolution_rate(u, t, boundary_mask, vector_array, deltas):
    laplacian_u = compute_laplacian(u, deltas)
    gradient_u = compute_gradient(u, deltas)
    divergence_u = compute_divergence(u, deltas)
    gradient_divergence_u = compute_gradient_scalar_field(divergence_u, deltas)
    
    # Convective term: (u · ∇)u
    convective_term = np.zeros_like(u)
    for i in prange(3):
        for j in range(3):
            convective_term[i] += u[j] * gradient_u[i, j]
    
    f_u = convective_term - shear_viscosity * laplacian_u - (bulk_viscosity + shear_viscosity / 3) * gradient_divergence_u
    ans = -f_u - 0.1 * vector_array
    
    # Apply boundary mask
    nx, ny, nz = ans.shape[1], ans.shape[2], ans.shape[3]
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if boundary_mask[i, j, k]:
                    ans[0, i, j, k] = 0.0
                    ans[1, i, j, k] = 0.0
                    ans[2, i, j, k] = 0.0
    return ans

@njit
def runge_kutta_4(u, t, dt, boundary_mask, vector_array, deltas):
    k1 = evolution_rate(u, t, boundary_mask, vector_array, deltas)
    k2 = evolution_rate(u + 0.5 * dt * k1, t + 0.5 * dt, boundary_mask, vector_array, deltas)
    k3 = evolution_rate(u + 0.5 * dt * k2, t + 0.5 * dt, boundary_mask, vector_array, deltas)
    k4 = evolution_rate(u + dt * k3, t + dt, boundary_mask, vector_array, deltas)
    u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next

# Main time-stepping loop
start_time = time.time()
t = 0.0
for step in range(num_steps):
    u = runge_kutta_4(u, t, dt, boundary_mask, vector_array, deltas)
    t += dt
    if step % 10 == 0:
        print(f"Time: {t:.2f} / {t_range}")

end_time = time.time()
print("Execution Time: ", end_time - start_time, " seconds")

# Plot of the magnitudes of the vector field at the intersection at x = 5
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(np.sqrt(u[0, X_SIZE//2, :, :]**2 + u[1, X_SIZE//2, :, :]**2 + u[2, X_SIZE//2, :, :]**2), origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('2D Vector Field Slice at X = 5')
plt.show()
