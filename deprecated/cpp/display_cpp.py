# Load the .csv file that contains the elements of the vector field stored in the following way
        # std::ofstream file("output.csv");
        # int scalar_size_local = nx_local * ny * nz;
        # for (int i_local = 0; i_local < nx_local; ++i_local) {
        #     for (int j = 0; j < ny; ++j) {
        #         for (int k = 0; k < nz; ++k) {
        #             int idx = index(i_local, j, k, ny, nz);
        #             file << u_local[idx] << "," << u_local[scalar_size_local + idx] << "," << u_local[2 * scalar_size_local + idx] << std::endl;
        #         }
        #     }
        # }

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = open("output.csv", "r")
lines = file.readlines()

X_SIZE = 160
Y_SIZE = 80
Z_SIZE = 80

# u = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
# x_grid = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
# y_grid = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
# z_grid = np.zeros((X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)

# for i, line in enumerate(lines):
#     x_grid[i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
#     y_grid[i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
#     z_grid[i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
#     u[0, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
#     u[1, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
#     u[2, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE] = map(float, line.split(","))
# file.close()


# u = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
# for i in range(X_SIZE):
#     for j in range(Y_SIZE):
#         for k in range(Z_SIZE):
#             u[0, i, j, k], u[1, i, j, k], u[2, i, j, k] = lines[i*Y_SIZE*Z_SIZE + j*Z_SIZE + k].split(",")

u = np.zeros((3, X_SIZE, Y_SIZE, Z_SIZE), dtype=np.float64)
for i, line in enumerate(lines):
    u[0, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
    u[1, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE], \
    u[2, i // (Y_SIZE * Z_SIZE), (i // Z_SIZE) % Y_SIZE, i % Z_SIZE] = map(float, line.split(","))


# Plot the magnitude of the cross section of the field at x = X_SIZE//2 (2d plot)
plt.figure()
plt.imshow(np.sqrt(u[0, X_SIZE//2]**2 + u[1, X_SIZE//2]**2 + u[2, X_SIZE//2]**2))
plt.colorbar()
plt.show()
