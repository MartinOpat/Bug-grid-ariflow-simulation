// pde_solver_mpi_corrected.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>
#include <fstream>

// Grid parameters
const int X_SIZE = 160;  // Should be divisible by the number of processes
const int Y_SIZE = 80;
const int Z_SIZE = 80;
const double X_RANGE[2] = {0.0, 10.0};
const double Y_RANGE[2] = {0.0, 5.0};
const double Z_RANGE[2] = {0.0, 5.0};

const double dx = (X_RANGE[1] - X_RANGE[0]) / X_SIZE;
const double dy = (Y_RANGE[1] - Y_RANGE[0]) / Y_SIZE;
const double dz = (Z_RANGE[1] - Z_RANGE[0]) / Z_SIZE;

// Time parameters
const double t_end = 1.0;
double dt = 0.01;  // Initial time step
const double dt_min = 1e-6;  // Minimum time step
const double dt_max = 0.1;   // Maximum time step

// Error tolerances
const double tol_abs = 1e-6;
const double tol_rel = 1e-3;
const double dt_safety = 0.9;

// Viscosity parameters
const double shear_viscosity = 0.1;
const double bulk_viscosity = 0.1;

// Function to create linearly spaced vector
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> vec(num);
    if (num == 1) {
        vec[0] = start;
    } else {
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            vec[i] = start + step * i;
        }
    }
    return vec;
}

// Helper function to compute index in flat array
inline int index(int i, int j, int k, int ny, int nz) {
    return i * ny * nz + j * nz + k;
}

// Evolution rate function
void evolution_rate(const std::vector<double>& u_local,
                    std::vector<double>& rate_local,
                    const std::vector<double>& vector_array_local,
                    const std::vector<bool>& boundary_mask_local,
                    int nx_local, int ny, int nz,
                    int rank, int size,
                    MPI_Comm comm) {

    int scalar_size_local = nx_local * ny * nz;
    int total_size_local = scalar_size_local * 3;  // 3 components

    // Initialize intermediate variables
    std::vector<double> laplacian_u(total_size_local, 0.0);
    std::vector<double> gradient_u_x(total_size_local, 0.0);
    std::vector<double> gradient_u_y(total_size_local, 0.0);
    std::vector<double> gradient_u_z(total_size_local, 0.0);

    // Exchange ghost cells for u_local (all components at once)
    int ny_nz = ny * nz;
    int num_components = 3;
    std::vector<double> send_left(ny_nz * num_components, 0.0);
    std::vector<double> send_right(ny_nz * num_components, 0.0);
    std::vector<double> recv_left(ny_nz * num_components, 0.0);
    std::vector<double> recv_right(ny_nz * num_components, 0.0);

    // Fill send buffers
    for (int comp = 0; comp < num_components; ++comp) {
        int offset = comp * scalar_size_local;
        for (int idx = 0; idx < ny_nz; ++idx) {
            send_left[comp * ny_nz + idx] = u_local[offset + idx];                                   // First layer
            send_right[comp * ny_nz + idx] = u_local[offset + (nx_local - 1) * ny_nz + idx];         // Last layer
        }
    }

    int rank_left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int rank_right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Exchange ghost cells
    MPI_Request requests[4];
    MPI_Isend(send_left.data(), ny_nz * num_components, MPI_DOUBLE, rank_left, 0, comm, &requests[0]);
    MPI_Isend(send_right.data(), ny_nz * num_components, MPI_DOUBLE, rank_right, 1, comm, &requests[1]);
    MPI_Irecv(recv_left.data(), ny_nz * num_components, MPI_DOUBLE, rank_left, 1, comm, &requests[2]);
    MPI_Irecv(recv_right.data(), ny_nz * num_components, MPI_DOUBLE, rank_right, 0, comm, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // Extend u_local with ghost cells
    int nx_ext = nx_local + 2;  // Including ghost cells
    int scalar_size_ext = nx_ext * ny * nz;
    std::vector<double> u_ext(scalar_size_ext * num_components, 0.0);

    // Copy inner data
    for (int comp = 0; comp < num_components; ++comp) {
        int offset = comp * scalar_size_local;
        int offset_ext = comp * scalar_size_ext;
        for (int i_local = 0; i_local < nx_local; ++i_local) {
            int i_ext = i_local + 1;
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    int idx_local = index(i_local, j, k, ny, nz);
                    int idx_ext = index(i_ext, j, k, ny, nz);
                    u_ext[offset_ext + idx_ext] = u_local[offset + idx_local];
                }
            }
        }
    }

    // Insert ghost cells
    for (int comp = 0; comp < num_components; ++comp) {
        int offset_ext = comp * scalar_size_ext;
        if (rank_left != MPI_PROC_NULL) {
            for (int idx = 0; idx < ny_nz; ++idx) {
                u_ext[offset_ext + idx] = recv_left[comp * ny_nz + idx];
            }
        } else {
            // Apply physical boundary condition at left physical boundary (Dirichlet: u = 0)
            for (int idx = 0; idx < ny_nz; ++idx) {
                u_ext[offset_ext + idx] = 0.0;
            }
        }
        if (rank_right != MPI_PROC_NULL) {
            for (int idx = 0; idx < ny_nz; ++idx) {
                u_ext[offset_ext + (nx_ext - 1) * ny_nz + idx] = recv_right[comp * ny_nz + idx];
            }
        } else {
            // Apply physical boundary condition at right physical boundary (Dirichlet: u = 0)
            for (int idx = 0; idx < ny_nz; ++idx) {
                u_ext[offset_ext + (nx_ext - 1) * ny_nz + idx] = 0.0;
            }
        }
    }

    // Compute gradient and laplacian for each component
    for (int comp = 0; comp < num_components; ++comp) {
        int offset_ext = comp * scalar_size_ext;
        int offset = comp * scalar_size_local;

        std::vector<double> dfdx(scalar_size_local, 0.0);
        std::vector<double> dfdy(scalar_size_local, 0.0);
        std::vector<double> dfdz(scalar_size_local, 0.0);
        std::vector<double> laplacian_f(scalar_size_local, 0.0);

        // Compute derivatives
        for (int i_local = 0; i_local < nx_local; ++i_local) {
            int i_ext = i_local + 1;
            for (int j = 0; j < ny; ++j) {
                int j_p = (j < ny - 1) ? j + 1 : j;  // Neumann boundary
                int j_m = (j > 0) ? j - 1 : j;       // Neumann boundary
                for (int k = 0; k < nz; ++k) {
                    int k_p = (k < nz - 1) ? k + 1 : k;  // Neumann boundary
                    int k_m = (k > 0) ? k - 1 : k;       // Neumann boundary

                    int idx = index(i_local, j, k, ny, nz);
                    int idx_ext = index(i_ext, j, k, ny, nz);
                    int idx_ip = index(i_ext + 1, j, k, ny, nz);
                    int idx_im = index(i_ext - 1, j, k, ny, nz);
                    int idx_jp = index(i_ext, j_p, k, ny, nz);
                    int idx_jm = index(i_ext, j_m, k, ny, nz);
                    int idx_kp = index(i_ext, j, k_p, ny, nz);
                    int idx_km = index(i_ext, j, k_m, ny, nz);

                    // Gradient
                    dfdx[idx] = (u_ext[offset_ext + idx_ip] - u_ext[offset_ext + idx_im]) / (2 * dx);
                    dfdy[idx] = (u_ext[offset_ext + idx_jp] - u_ext[offset_ext + idx_jm]) / (2 * dy);
                    dfdz[idx] = (u_ext[offset_ext + idx_kp] - u_ext[offset_ext + idx_km]) / (2 * dz);

                    // Laplacian
                    double d2fdx2 = (u_ext[offset_ext + idx_ip] - 2 * u_ext[offset_ext + idx_ext] + u_ext[offset_ext + idx_im]) / (dx * dx);
                    double d2fdy2 = (u_ext[offset_ext + idx_jp] - 2 * u_ext[offset_ext + idx_ext] + u_ext[offset_ext + idx_jm]) / (dy * dy);
                    double d2fdz2 = (u_ext[offset_ext + idx_kp] - 2 * u_ext[offset_ext + idx_ext] + u_ext[offset_ext + idx_km]) / (dz * dz);
                    laplacian_f[idx] = d2fdx2 + d2fdy2 + d2fdz2;
                }
            }
        }

        // Store gradients and laplacian
        for (int idx = 0; idx < scalar_size_local; ++idx) {
            gradient_u_x[offset + idx] = dfdx[idx];
            gradient_u_y[offset + idx] = dfdy[idx];
            gradient_u_z[offset + idx] = dfdz[idx];
            laplacian_u[offset + idx] = laplacian_f[idx];
        }
    }

    // Compute divergence of u
    std::vector<double> divergence_u(scalar_size_local, 0.0);
    for (int idx = 0; idx < scalar_size_local; ++idx) {
        divergence_u[idx] = gradient_u_x[0 * scalar_size_local + idx] + gradient_u_y[1 * scalar_size_local + idx] + gradient_u_z[2 * scalar_size_local + idx];
    }

    // Exchange ghost cells for divergence_u
    std::vector<double> send_left_div(ny_nz, 0.0);
    std::vector<double> send_right_div(ny_nz, 0.0);
    std::vector<double> recv_left_div(ny_nz, 0.0);
    std::vector<double> recv_right_div(ny_nz, 0.0);

    // Fill send buffers
    for (int idx = 0; idx < ny_nz; ++idx) {
        send_left_div[idx] = divergence_u[idx];                                   // First layer
        send_right_div[idx] = divergence_u[(nx_local - 1) * ny_nz + idx];         // Last layer
    }

    // Exchange ghost cells for divergence_u
    MPI_Request requests_div[4];
    MPI_Isend(send_left_div.data(), ny_nz, MPI_DOUBLE, rank_left, 2, comm, &requests_div[0]);
    MPI_Isend(send_right_div.data(), ny_nz, MPI_DOUBLE, rank_right, 3, comm, &requests_div[1]);
    MPI_Irecv(recv_left_div.data(), ny_nz, MPI_DOUBLE, rank_left, 3, comm, &requests_div[2]);
    MPI_Irecv(recv_right_div.data(), ny_nz, MPI_DOUBLE, rank_right, 2, comm, &requests_div[3]);
    MPI_Waitall(4, requests_div, MPI_STATUSES_IGNORE);

    // Extend divergence_u with ghost cells
    int nx_ext_div = nx_local + 2;  // Including ghost cells
    std::vector<double> divergence_u_ext(nx_ext_div * ny * nz, 0.0);

    // Copy inner data
    for (int i_local = 0; i_local < nx_local; ++i_local) {
        int i_ext = i_local + 1;
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx_local = index(i_local, j, k, ny, nz);
                int idx_ext = index(i_ext, j, k, ny, nz);
                divergence_u_ext[idx_ext] = divergence_u[idx_local];
            }
        }
    }

    // Insert ghost cells
    if (rank_left != MPI_PROC_NULL) {
        for (int idx = 0; idx < ny_nz; ++idx) {
            divergence_u_ext[idx] = recv_left_div[idx];
        }
    } else {
        // Apply physical boundary condition at left physical boundary (Neumann: derivative = 0)
        for (int idx = 0; idx < ny_nz; ++idx) {
            divergence_u_ext[idx] = divergence_u_ext[ny_nz + idx];  // Copy first inner layer
        }
    }
    if (rank_right != MPI_PROC_NULL) {
        for (int idx = 0; idx < ny_nz; ++idx) {
            divergence_u_ext[(nx_ext_div - 1) * ny_nz + idx] = recv_right_div[idx];
        }
    } else {
        // Apply physical boundary condition at right physical boundary (Neumann: derivative = 0)
        for (int idx = 0; idx < ny_nz; ++idx) {
            divergence_u_ext[(nx_ext_div - 1) * ny_nz + idx] = divergence_u_ext[(nx_ext_div - 2) * ny_nz + idx];  // Copy last inner layer
        }
    }

    // Compute gradient of divergence_u
    std::vector<double> grad_div_x(scalar_size_local, 0.0);
    std::vector<double> grad_div_y(scalar_size_local, 0.0);
    std::vector<double> grad_div_z(scalar_size_local, 0.0);

    for (int i_local = 0; i_local < nx_local; ++i_local) {
        int i_ext = i_local + 1;
        for (int j = 0; j < ny; ++j) {
            int j_p = (j < ny - 1) ? j + 1 : j;
            int j_m = (j > 0) ? j - 1 : j;
            for (int k = 0; k < nz; ++k) {
                int k_p = (k < nz - 1) ? k + 1 : k;
                int k_m = (k > 0) ? k - 1 : k;

                int idx = index(i_local, j, k, ny, nz);
                int idx_ext = index(i_ext, j, k, ny, nz);
                int idx_ip = index(i_ext + 1, j, k, ny, nz);
                int idx_im = index(i_ext - 1, j, k, ny, nz);
                int idx_jp = index(i_ext, j_p, k, ny, nz);
                int idx_jm = index(i_ext, j_m, k, ny, nz);
                int idx_kp = index(i_ext, j, k_p, ny, nz);
                int idx_km = index(i_ext, j, k_m, ny, nz);

                grad_div_x[idx] = (divergence_u_ext[idx_ip] - divergence_u_ext[idx_im]) / (2 * dx);
                grad_div_y[idx] = (divergence_u_ext[idx_jp] - divergence_u_ext[idx_jm]) / (2 * dy);
                grad_div_z[idx] = (divergence_u_ext[idx_kp] - divergence_u_ext[idx_km]) / (2 * dz);
            }
        }
    }

    // Compute convective term and rate
    for (int idx = 0; idx < scalar_size_local; ++idx) {
        for (int comp = 0; comp < 3; ++comp) {
            int idx_comp = comp * scalar_size_local + idx;

            double u_dot_grad_u = u_local[0 * scalar_size_local + idx] * gradient_u_x[idx_comp]
                                + u_local[1 * scalar_size_local + idx] * gradient_u_y[idx_comp]
                                + u_local[2 * scalar_size_local + idx] * gradient_u_z[idx_comp];

            double grad_div = (comp == 0) ? grad_div_x[idx] : (comp == 1) ? grad_div_y[idx] : grad_div_z[idx];

            double f_u = u_dot_grad_u
                - shear_viscosity * laplacian_u[idx_comp]
                - (bulk_viscosity + shear_viscosity / 3.0) * grad_div;

            rate_local[idx_comp] = -f_u - 0.1 * vector_array_local[idx_comp];

            // Apply boundary mask
            if (boundary_mask_local[idx]) {
                rate_local[idx_comp] = 0.0;
            }
        }
    }
}

// Adaptive Runge-Kutta method
void adaptive_runge_kutta(std::vector<double>& u_local, double& t, double& dt,
                          const std::vector<double>& vector_array_local,
                          const std::vector<bool>& boundary_mask_local,
                          int nx_local, int ny, int nz,
                          int rank, int size, MPI_Comm comm) {

    int total_size_local = nx_local * ny * nz * 3;  // 3 components

    bool step_accepted = false;
    double dt_new = dt;

    // Temporary variables
    std::vector<double> u_temp(total_size_local, 0.0);
    std::vector<double> k1(total_size_local, 0.0);
    std::vector<double> k2(total_size_local, 0.0);
    std::vector<double> k3(total_size_local, 0.0);
    std::vector<double> k4(total_size_local, 0.0);
    std::vector<double> k5(total_size_local, 0.0);

    while (!step_accepted) {
        // Compute k1
        evolution_rate(u_local, k1, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, comm);

        // Compute k2
        for (int idx = 0; idx < total_size_local; ++idx)
            u_temp[idx] = u_local[idx] + 0.25 * dt * k1[idx];
        evolution_rate(u_temp, k2, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, comm);

        // Compute k3
        for (int idx = 0; idx < total_size_local; ++idx)
            u_temp[idx] = u_local[idx] + (3.0 / 32.0) * dt * k1[idx] + (9.0 / 32.0) * dt * k2[idx];
        evolution_rate(u_temp, k3, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, comm);

        // Compute k4
        for (int idx = 0; idx < total_size_local; ++idx)
            u_temp[idx] = u_local[idx] + (1932.0 / 2197.0) * dt * k1[idx]
                                  - (7200.0 / 2197.0) * dt * k2[idx]
                                  + (7296.0 / 2197.0) * dt * k3[idx];
        evolution_rate(u_temp, k4, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, comm);

        // Compute k5
        for (int idx = 0; idx < total_size_local; ++idx)
            u_temp[idx] = u_local[idx] + (439.0 / 216.0) * dt * k1[idx]
                                  - 8.0 * dt * k2[idx]
                                  + (3680.0 / 513.0) * dt * k3[idx]
                                  - (845.0 / 4104.0) * dt * k4[idx];
        evolution_rate(u_temp, k5, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, comm);

        // Compute 4th order solution
        std::vector<double> u4th(total_size_local, 0.0);
        for (int idx = 0; idx < total_size_local; ++idx)
            u4th[idx] = u_local[idx] + dt * (
                (25.0 / 216.0) * k1[idx]
                + (1408.0 / 2565.0) * k3[idx]
                + (2197.0 / 4104.0) * k4[idx]
                - (1.0 / 5.0) * k5[idx]);

        // Compute 5th order solution
        std::vector<double> u5th(total_size_local, 0.0);
        for (int idx = 0; idx < total_size_local; ++idx)
            u5th[idx] = u_local[idx] + dt * (
                (16.0 / 135.0) * k1[idx]
                + (6656.0 / 12825.0) * k3[idx]
                + (28561.0 / 56430.0) * k4[idx]
                - (9.0 / 50.0) * k5[idx]
                + (2.0 / 55.0) * k5[idx]);

        // Estimate error
        double local_max_error = 0.0;
        for (int idx = 0; idx < total_size_local; ++idx) {
            double err = std::abs(u5th[idx] - u4th[idx]);
            double tol = tol_abs + tol_rel * std::max(std::abs(u4th[idx]), std::abs(u5th[idx]));
            double norm_err = err / tol;
            if (norm_err > local_max_error) {
                local_max_error = norm_err;
            }
        }

        // Global maximum error
        double global_max_error = 0.0;
        MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, comm);

        // Decide if step is accepted
        if (global_max_error <= 1.0) {
            // Step is accepted
            step_accepted = true;
            u_local = u5th;
            t += dt;

            // Compute new dt
            dt_new = dt * std::min(dt_safety * std::pow(global_max_error, -0.2), 5.0);
            dt_new = std::min(dt_new, dt_max);
        } else {
            // Step is rejected, decrease dt
            dt_new = dt * std::max(dt_safety * std::pow(global_max_error, -0.25), 0.1);
            dt_new = std::max(dt_new, dt_min);
            // Ensure dt does not become too small
            if (dt_new < dt_min) {
                if (rank == 0) {
                    std::cerr << "Error: Time step below minimum allowed value." << std::endl;
                }
                MPI_Abort(comm, 1);
            }
        }

        // All processes must agree on dt
        MPI_Bcast(&dt_new, 1, MPI_DOUBLE, 0, comm);
        dt = dt_new;

        // Synchronize time t across processes
        MPI_Bcast(&t, 1, MPI_DOUBLE, 0, comm);
    }
}

// Main function
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace std;

    // Ensure that X_SIZE is divisible by size
    if (X_SIZE % size != 0) {
        if (rank == 0) {
            cerr << "Error: X_SIZE must be divisible by the number of processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int nx_local = X_SIZE / size;
    int ny = Y_SIZE;
    int nz = Z_SIZE;

    int scalar_size_local = nx_local * ny * nz;
    int total_size_local = scalar_size_local * 3;  // For 3 components

    // Create local grid indices
    int x_start = rank * nx_local;
    int x_end = x_start + nx_local;

    vector<double> x = linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, X_SIZE);
    vector<double> y = linspace(Y_RANGE[0] + dy / 2, Y_RANGE[1] - dy / 2, ny);
    vector<double> z = linspace(Z_RANGE[0] + dz / 2, Z_RANGE[1] - dz / 2, nz);

    // Initialize local grid matrices
    vector<double> x_grid_local(scalar_size_local, 0.0);
    vector<double> y_grid_local(scalar_size_local, 0.0);
    vector<double> z_grid_local(scalar_size_local, 0.0);

    for (int i_local = 0; i_local < nx_local; ++i_local) {
        int i_global = x_start + i_local;
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx = index(i_local, j, k, ny, nz);
                x_grid_local[idx] = x[i_global];
                y_grid_local[idx] = y[j];
                z_grid_local[idx] = z[k];
            }
        }
    }

    // Initialize fields
    vector<double> u_local(total_size_local, 0.0);
    vector<double> vector_array_local(total_size_local, 0.0);

    int center = X_SIZE / 2;

    // **Corrected initialization of vector_array_local**
    for (int i_local = 0; i_local < nx_local; ++i_local) {
        int i_global = x_start + i_local;
        if (i_global == center - 1 || i_global == center) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    int idx = index(i_local, j, k, ny, nz);
                    for (int comp = 0; comp < 3; ++comp) {
                        vector_array_local[comp * scalar_size_local + idx] = 1.0;
                    }
                }
            }
        }
    }

    // Define boundary mask
    double y_width = 0.5;
    int y_count = 5;
    double z_width = 0.5;
    int z_count = 5;

    vector<bool> boundary_mask_local(scalar_size_local, false);

    for (int i_local = 0; i_local < nx_local; ++i_local) {
        int i_global = x_start + i_local;
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx = index(i_local, j, k, ny, nz);
                bool in_x_range = (x_grid_local[idx] >= 3.0) && (x_grid_local[idx] <= 7.0);
                double y_mod = fmod(y_grid_local[idx], 5.0 / y_count);
                double z_mod = fmod(z_grid_local[idx], 5.0 / z_count);
                bool y_condition = (y_mod >= (5.0 / y_count - y_width / 2)) || (y_mod <= y_width / 2);
                bool z_condition = (z_mod >= (5.0 / z_count - z_width / 2)) || (z_mod <= z_width / 2);
                if (in_x_range && (y_condition || z_condition)) {
                    boundary_mask_local[idx] = true;
                }
            }
        }
    }

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Time-stepping loop
    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        adaptive_runge_kutta(u_local, t, dt, vector_array_local, boundary_mask_local, nx_local, ny, nz, rank, size, MPI_COMM_WORLD);
        step++;
        if (step % 10 == 0 && rank == 0) {
            std::cout << "Time: " << t << " / " << t_end << ", dt: " << dt << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    if (rank == 0) {
        std::cout << "Execution Time: " << execution_time.count() << " seconds" << std::endl;

        // Write the results into a .csv file
        std::ofstream file("results.csv");
        for (int i_local = 0; i_local < nx_local; ++i_local) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    int idx = index(i_local, j, k, ny, nz);
                    file << x_grid_local[idx] << "," << y_grid_local[idx] << "," << z_grid_local[idx];
                    for (int comp = 0; comp < 3; ++comp) {
                        file << "," << u_local[comp * scalar_size_local + idx];
                    }
                    file << std::endl;
                }
            }
        }
        file.close();
    }

    MPI_Finalize();
    return 0;
}
