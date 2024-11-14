import numpy as np
import time

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

# Shear and bulk viscosity
shear_viscosity = 0.1
bulk_viscosity = 0.1

# Time parameters
t_range = 0.5
dt = 0.01
num_steps = int(t_range / dt)

# Helper functions for finite differences
def derivative(f, axis, delta):
    df = np.zeros_like(f)
    # Central differences for interior points
    slicer_center = [slice(None)] * 3
    slicer_plus = [slice(None)] * 3
    slicer_minus = [slice(None)] * 3
    slicer_center[axis] = slice(1, -1)
    slicer_plus[axis] = slice(2, None)
    slicer_minus[axis] = slice(None, -2)
    df[tuple(slicer_center)] = (f[tuple(slicer_plus)] - f[tuple(slicer_minus)]) / (2 * delta)
    # Forward difference at the first point
    slicer_first = [slice(None)] * 3
    slicer_first[axis] = 0
    slicer_next = [slice(None)] * 3
    slicer_next[axis] = 1
    df[tuple(slicer_first)] = (f[tuple(slicer_next)] - f[tuple(slicer_first)]) / delta
    # Backward difference at the last point
    slicer_last = [slice(None)] * 3
    slicer_last[axis] = -1
    slicer_prev = [slice(None)] * 3
    slicer_prev[axis] = -2
    df[tuple(slicer_last)] = (f[tuple(slicer_last)] - f[tuple(slicer_prev)]) / delta
    return df

def second_derivative(f, axis, delta):
    d2f = np.zeros_like(f)
    # Second differences for interior points
    slicer_center = [slice(None)] * 3
    slicer_plus = [slice(None)] * 3
    slicer_minus = [slice(None)] * 3
    slicer_center[axis] = slice(1, -1)
    slicer_plus[axis] = slice(2, None)
    slicer_minus[axis] = slice(None, -2)
    d2f[tuple(slicer_center)] = (f[tuple(slicer_plus)] - 2 * f[tuple(slicer_center)] + f[tuple(slicer_minus)]) / (delta ** 2)
    # Boundary points
    slicer_first = [slice(None)] * 3
    slicer_first[axis] = 0
    slicer_next = [slice(None)] * 3
    slicer_next[axis] = 1
    d2f[tuple(slicer_first)] = (f[tuple(slicer_next)] - 2 * f[tuple(slicer_first)]) / (delta ** 2)
    slicer_last = [slice(None)] * 3
    slicer_last[axis] = -1
    slicer_prev = [slice(None)] * 3
    slicer_prev[axis] = -2
    d2f[tuple(slicer_last)] = (-2 * f[tuple(slicer_last)] + f[tuple(slicer_prev)]) / (delta ** 2)
    return d2f

def compute_gradient(u, dx, dy, dz):
    gradient_u = np.zeros((3, 3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    deltas = [dx, dy, dz]
    for i in range(3):  # Components of u
        for j in range(3):  # Derivative with respect to x_j
            gradient_u[i, j] = derivative(u[i], axis=j, delta=deltas[j])
    return gradient_u

def compute_divergence(u, dx, dy, dz):
    divergence_u = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    deltas = [dx, dy, dz]
    for i in range(3):  # Components of u
        divergence_u += derivative(u[i], axis=i, delta=deltas[i])
    return divergence_u

def compute_gradient_scalar_field(f, dx, dy, dz):
    grad_f = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE))
    deltas = [dx, dy, dz]
    for i in range(3):
        grad_f[i] = derivative(f, axis=i, delta=deltas[i])
    return grad_f

def compute_laplacian(u, dx, dy, dz):
    laplacian_u = np.zeros_like(u)
    deltas = [dx, dy, dz]
    for i in range(3):  # Components of u
        for axis in range(3):
            laplacian_u[i] += second_derivative(u[i], axis=axis, delta=deltas[axis])
    return laplacian_u

# Evolution rate function
def evolution_rate(u, t):
    laplacian_u = compute_laplacian(u, dx, dy, dz)
    gradient_u = compute_gradient(u, dx, dy, dz)
    divergence_u = compute_divergence(u, dx, dy, dz)
    gradient_divergence_u = compute_gradient_scalar_field(divergence_u, dx, dy, dz)
    
    # Convective term: (u · ∇)u
    convective_term = np.zeros_like(u)
    for i in range(3):
        for j in range(3):
            convective_term[i] += u[j] * gradient_u[i, j]
    
    f_u = convective_term - shear_viscosity * laplacian_u - (bulk_viscosity + shear_viscosity / 3) * gradient_divergence_u
    ans = -f_u - 0.1 * vector_array
    # Apply boundary mask
    ans[:, boundary_mask] = 0
    return ans

# Time integration using Runge-Kutta 4
def runge_kutta_4(u, t, dt):
    k1 = evolution_rate(u, t)
    k2 = evolution_rate(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = evolution_rate(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = evolution_rate(u + dt * k3, t + dt)
    u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next

# Main time-stepping loop
start_time = time.time()
t = 0.0
for step in range(num_steps):
    u = runge_kutta_4(u, t, dt)
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

