import numpy as np
import time
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Grid parameters
X_SIZE_TOTAL = 160  # Total size along X
Y_SIZE = 80
Z_SIZE = 80
X_RANGE = (0, 10)
Y_RANGE = (0, 5)
Z_RANGE = (0, 5)

dx = (X_RANGE[1] - X_RANGE[0]) / X_SIZE_TOTAL
dy = (Y_RANGE[1] - Y_RANGE[0]) / Y_SIZE
dz = (Z_RANGE[1] - Z_RANGE[0]) / Z_SIZE
deltas = np.array([dx, dy, dz])

# Divide the grid among MPI processes along the X-axis
X_SIZES = [X_SIZE_TOTAL // size] * size
for i in range(X_SIZE_TOTAL % size):
    X_SIZES[i] += 1

X_STARTS = [sum(X_SIZES[:i]) for i in range(size)]
X_SIZE = X_SIZES[rank]
X_START = X_STARTS[rank]

# Local grid coordinates
x_local = np.linspace(
    X_RANGE[0] + (X_START + 0.5) * dx,
    X_RANGE[0] + (X_START + X_SIZE - 0.5) * dx,
    X_SIZE
)
y = np.linspace(Y_RANGE[0] + dy/2, Y_RANGE[1] - dy/2, Y_SIZE)
z = np.linspace(Z_RANGE[0] + dz/2, Z_RANGE[1] - dz/2, Z_SIZE)

x_grid, y_grid, z_grid = np.meshgrid(x_local, y, z, indexing='ij')

# Initialize fields
u_local = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
vector_array_local = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)

# Initialize vector_array
center_x = X_SIZE_TOTAL // 2
if X_START <= center_x < X_START + X_SIZE:
    local_index = center_x - X_START
    vector_array_local[0, local_index - 1:local_index + 2, :, :] = 1  # Set x-component to 1 at the center slice

# Boundary mask
y_width = 0.5
y_count = 5
z_width = 0.5
z_count = 5
boundary_mask_local = (
    ((3 <= x_grid) & (x_grid <= 7)) &
    (
        ((y_grid % (5 / y_count) >= 5 / y_count - y_width / 2) | (y_grid % (5 / y_count) <= y_width / 2)) |
        ((z_grid % (5 / z_count) >= 5 / z_count - z_width / 2) | (z_grid % (5 / z_count) <= z_width / 2))
    )
)
boundary_mask_local = boundary_mask_local.astype(np.bool_)

# Shear and bulk viscosity
shear_viscosity = 0.1
bulk_viscosity = 0.1

# Time parameters
t_range = 1.0
dt = 0.01
num_steps = int(t_range / dt)

# Helper functions for finite differences
def exchange_ghost_cells(f_local):
    # Communicate ghost cells with neighbors along the X-axis
    send_left = f_local[:, 0, :, :].copy()
    send_right = f_local[:, -1, :, :].copy()
    recv_left = np.zeros_like(send_left)
    recv_right = np.zeros_like(send_right)
    left_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    req = []
    # Send to left, receive from right
    req.append(comm.Isend(send_left, dest=left_rank, tag=10))
    req.append(comm.Irecv(recv_right, source=right_rank, tag=10))
    # Send to right, receive from left
    req.append(comm.Isend(send_right, dest=right_rank, tag=11))
    req.append(comm.Irecv(recv_left, source=left_rank, tag=11))
    MPI.Request.Waitall(req)
    return recv_left, recv_right

def exchange_ghost_cells_scalar(f_local):
    # Communicate ghost cells with neighbors along the X-axis for scalar fields
    send_left = f_local[0, :, :].copy()
    send_right = f_local[-1, :, :].copy()
    recv_left = np.zeros_like(send_left)
    recv_right = np.zeros_like(send_right)
    left_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    req = []
    # Send to left, receive from right
    req.append(comm.Isend(send_left, dest=left_rank, tag=20))
    req.append(comm.Irecv(recv_right, source=right_rank, tag=20))
    # Send to right, receive from left
    req.append(comm.Isend(send_right, dest=right_rank, tag=21))
    req.append(comm.Irecv(recv_left, source=left_rank, tag=21))
    MPI.Request.Waitall(req)
    return recv_left, recv_right

def derivative(f_local, delta, axis):
    if axis == 0:
        # Exchange ghost cells along X-axis
        recv_left, recv_right = exchange_ghost_cells(f_local)
        f_with_ghosts = np.pad(f_local, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        f_with_ghosts[:, 0, :, :] = recv_left
        f_with_ghosts[:, -1, :, :] = recv_right
        df = (f_with_ghosts[:, 2:, :, :] - f_with_ghosts[:, :-2, :, :]) / (2 * delta)
    elif axis == 1:
        f_with_ghosts = np.pad(f_local, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
        df = (f_with_ghosts[:, :, 2:, :] - f_with_ghosts[:, :, :-2, :]) / (2 * delta)
    else:
        f_with_ghosts = np.pad(f_local, ((0, 0), (0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
        df = (f_with_ghosts[:, :, :, 2:] - f_with_ghosts[:, :, :, :-2]) / (2 * delta)
    return df

def derivative_scalar(f_local, delta, axis):
    if axis == 0:
        # Exchange ghost cells along X-axis
        recv_left, recv_right = exchange_ghost_cells_scalar(f_local)
        f_with_ghosts = np.pad(f_local, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        f_with_ghosts[0, :, :] = recv_left
        f_with_ghosts[-1, :, :] = recv_right
        df = (f_with_ghosts[2:, :, :] - f_with_ghosts[:-2, :, :]) / (2 * delta)
    elif axis == 1:
        f_with_ghosts = np.pad(f_local, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
        df = (f_with_ghosts[:, 2:, :] - f_with_ghosts[:, :-2, :]) / (2 * delta)
    else:
        f_with_ghosts = np.pad(f_local, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
        df = (f_with_ghosts[:, :, 2:] - f_with_ghosts[:, :, :-2]) / (2 * delta)
    return df

def second_derivative(f_local, delta, axis):
    if axis == 0:
        # Exchange ghost cells along X-axis
        recv_left, recv_right = exchange_ghost_cells_scalar(f_local)
        f_with_ghosts = np.pad(f_local, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        f_with_ghosts[0, :, :] = recv_left
        f_with_ghosts[-1, :, :] = recv_right
        d2f = (f_with_ghosts[2:, :, :] - 2 * f_with_ghosts[1:-1, :, :] + f_with_ghosts[:-2, :, :]) / (delta ** 2)
    elif axis == 1:
        f_with_ghosts = np.pad(f_local, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
        d2f = (f_with_ghosts[:, 2:, :] - 2 * f_with_ghosts[:, 1:-1, :] + f_with_ghosts[:, :-2, :]) / (delta ** 2)
    else:
        f_with_ghosts = np.pad(f_local, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
        d2f = (f_with_ghosts[:, :, 2:] - 2 * f_with_ghosts[:, :, 1:-1] + f_with_ghosts[:, :, :-2]) / (delta ** 2)
    return d2f

def compute_gradient(u_local, deltas):
    gradient_u = np.zeros((3, 3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for comp in range(3):
        for axis in range(3):
            gradient_u[comp, axis] = derivative(u_local[comp:comp+1], deltas[axis], axis)[0]
    return gradient_u

def compute_divergence(u_local, deltas):
    divergence_u = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for axis in range(3):
        du = derivative(u_local[axis:axis+1], deltas[axis], axis)[0]
        divergence_u += du
    return divergence_u

def compute_gradient_of_divergence(divergence_u_local, deltas):
    grad_div_u = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
    for axis in range(3):
        grad_div_u[axis] = derivative_scalar(divergence_u_local, deltas[axis], axis)
    return grad_div_u

def compute_laplacian(u_local, deltas):
    laplacian_u = np.zeros_like(u_local)
    for comp in range(3):
        for axis in range(3):
            laplacian_u[comp] += second_derivative(u_local[comp], deltas[axis], axis)
    return laplacian_u

def evolution_rate(u_local, t):
    laplacian_u = compute_laplacian(u_local, deltas)
    gradient_u = compute_gradient(u_local, deltas)
    divergence_u = compute_divergence(u_local, deltas)
    gradient_divergence_u = compute_gradient_of_divergence(divergence_u, deltas)

    # Convective term: (u · ∇)u
    convective_term = np.zeros_like(u_local)
    for i in range(3):
        for j in range(3):
            convective_term[i] += u_local[j] * gradient_u[i, j]

    f_u = convective_term - shear_viscosity * laplacian_u - (bulk_viscosity + shear_viscosity / 3) * gradient_divergence_u
    ans = -f_u - 0.1 * vector_array_local

    # Apply boundary mask
    ans[:, boundary_mask_local] = 0
    return ans

# Runge-Kutta 4 method
def runge_kutta_4(u_local, t, dt):
    k1 = evolution_rate(u_local, t)
    u_temp = u_local + 0.5 * dt * k1
    k2 = evolution_rate(u_temp, t + 0.5 * dt)
    u_temp = u_local + 0.5 * dt * k2
    k3 = evolution_rate(u_temp, t + 0.5 * dt)
    u_temp = u_local + dt * k3
    k4 = evolution_rate(u_temp, t + dt)
    u_next = u_local + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_next

# Main time-stepping loop
start_time = time.time()
t = 0.0
for step in range(num_steps):
    u_local = runge_kutta_4(u_local, t, dt)
    t += dt
    if step % 10 == 0 and rank == 0:
        print(f"Time: {t:.2f} / {t_range}")

end_time = time.time()
if rank == 0:
    print("Execution Time: ", end_time - start_time, " seconds")

# Gather results from all processes to assemble the global u
# Only the root process (rank 0) will have the assembled u
u_global = None
if rank == 0:
    u_global = np.empty((3, X_SIZE_TOTAL, Y_SIZE, Z_SIZE), dtype=np.float64)

# Gather u_local arrays from all processes
u_local_flat = u_local.ravel()
u_counts = comm.gather(u_local_flat.size, root=0)

if rank == 0:
    u_displs = np.insert(np.cumsum(u_counts[:-1]), 0, 0)
else:
    u_displs = None

comm.Gatherv(sendbuf=u_local_flat, recvbuf=(u_global.ravel() if rank == 0 else None, (u_counts, u_displs)), root=0)

import matplotlib.pyplot as plt
if rank == 0:
    print("Final result collected in u_global with shape:", u_global.shape)


    # Plot a slice of the x-component of the velocity field
    plt.imshow(u_global[0, X_SIZE_TOTAL // 2, :, :])
    plt.colorbar()
    plt.show()

