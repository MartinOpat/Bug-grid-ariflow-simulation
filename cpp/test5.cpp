#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <fstream>

// Grid parameters
const int X_SIZE = 160;  // Should be divisible by 2
const int Y_SIZE = 80;
const int Z_SIZE = 80;
const int nx = X_SIZE;
const int ny = Y_SIZE;
const int nz = Z_SIZE;

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

// Helper function to compute index in flat array
inline int index(int i, int j, int k, int nx, int ny, int nz) {
    return i * ny * nz + j * nz + k;
}

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

// Helper functions for finite differences
void compute_gradient(const std::vector<double>& f,
                      std::vector<double>& dfdx,
                      std::vector<double>& dfdy,
                      std::vector<double>& dfdz,
                      double dx, double dy, double dz,
                      int nx, int ny, int nz) {

    #pragma omp parallel
    {
        // Compute dfdx
        #pragma omp for
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    int idx = index(i, j, k, nx, ny, nz);
                    int idx_forward = index(i + 1, j, k, nx, ny, nz);
                    int idx_backward = index(i - 1, j, k, nx, ny, nz);
                    dfdx[idx] = (f[idx_forward] - f[idx_backward]) / (2 * dx);
                }
            }
        }

        // Boundary conditions for dfdx
        #pragma omp for
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx0 = index(0, j, k, nx, ny, nz);
                int idx1 = index(1, j, k, nx, ny, nz);
                int idx_end = index(nx - 1, j, k, nx, ny, nz);
                int idx_endm1 = index(nx - 2, j, k, nx, ny, nz);
                dfdx[idx0] = (f[idx1] - f[idx0]) / dx;
                dfdx[idx_end] = (f[idx_end] - f[idx_endm1]) / dx;
            }
        }

        // Compute dfdy
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int k = 0; k < nz; ++k) {
                    int idx = index(i, j, k, nx, ny, nz);
                    int idx_forward = index(i, j + 1, k, nx, ny, nz);
                    int idx_backward = index(i, j - 1, k, nx, ny, nz);
                    dfdy[idx] = (f[idx_forward] - f[idx_backward]) / (2 * dy);
                }
            }
        }

        // Boundary conditions for dfdy
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int k = 0; k < nz; ++k) {
                int idx0 = index(i, 0, k, nx, ny, nz);
                int idx1 = index(i, 1, k, nx, ny, nz);
                int idx_end = index(i, ny - 1, k, nx, ny, nz);
                int idx_endm1 = index(i, ny - 2, k, nx, ny, nz);
                dfdy[idx0] = (f[idx1] - f[idx0]) / dy;
                dfdy[idx_end] = (f[idx_end] - f[idx_endm1]) / dy;
            }
        }

        // Compute dfdz
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 1; k < nz - 1; ++k) {
                    int idx = index(i, j, k, nx, ny, nz);
                    int idx_forward = index(i, j, k + 1, nx, ny, nz);
                    int idx_backward = index(i, j, k - 1, nx, ny, nz);
                    dfdz[idx] = (f[idx_forward] - f[idx_backward]) / (2 * dz);
                }
            }
        }

        // Boundary conditions for dfdz
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int idx0 = index(i, j, 0, nx, ny, nz);
                int idx1 = index(i, j, 1, nx, ny, nz);
                int idx_end = index(i, j, nz - 1, nx, ny, nz);
                int idx_endm1 = index(i, j, nz - 2, nx, ny, nz);
                dfdz[idx0] = (f[idx1] - f[idx0]) / dz;
                dfdz[idx_end] = (f[idx_end] - f[idx_endm1]) / dz;
            }
        }
    }
}

void compute_laplacian(const std::vector<double>& f,
                       std::vector<double>& laplacian_f,
                       double dx, double dy, double dz,
                       int nx, int ny, int nz) {

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int k = 1; k < nz - 1; ++k) {
                    int idx = index(i, j, k, nx, ny, nz);
                    int idx_ip = index(i + 1, j, k, nx, ny, nz);
                    int idx_im = index(i - 1, j, k, nx, ny, nz);
                    int idx_jp = index(i, j + 1, k, nx, ny, nz);
                    int idx_jm = index(i, j - 1, k, nx, ny, nz);
                    int idx_kp = index(i, j, k + 1, nx, ny, nz);
                    int idx_km = index(i, j, k - 1, nx, ny, nz);

                    double d2fdx2 = (f[idx_ip] - 2 * f[idx] + f[idx_im]) / (dx * dx);
                    double d2fdy2 = (f[idx_jp] - 2 * f[idx] + f[idx_jm]) / (dy * dy);
                    double d2fdz2 = (f[idx_kp] - 2 * f[idx] + f[idx_km]) / (dz * dz);
                    laplacian_f[idx] = d2fdx2 + d2fdy2 + d2fdz2;
                }
            }
        }

        // Boundary conditions: set laplacian to zero at boundaries
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                laplacian_f[index(i, j, 0, nx, ny, nz)] = 0.0;
                laplacian_f[index(i, j, nz - 1, nx, ny, nz)] = 0.0;
            }
        }
        #pragma omp for
        for (int i = 0; i < nx; ++i) {
            for (int k = 0; k < nz; ++k) {
                laplacian_f[index(i, 0, k, nx, ny, nz)] = 0.0;
                laplacian_f[index(i, ny - 1, k, nx, ny, nz)] = 0.0;
            }
        }
        #pragma omp for
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                laplacian_f[index(0, j, k, nx, ny, nz)] = 0.0;
                laplacian_f[index(nx - 1, j, k, nx, ny, nz)] = 0.0;
            }
        }
    }
}

// Evolution rate function
void evolution_rate(const std::vector<double>& u,
                    std::vector<double>& rate,
                    const std::vector<double>& vector_array,
                    const std::vector<bool>& boundary_mask,
                    int nx, int ny, int nz) {

    // Initialize intermediate variables
    std::vector<double> laplacian_u(u.size(), 0.0);
    std::vector<double> gradient_u_x(u.size(), 0.0);
    std::vector<double> gradient_u_y(u.size(), 0.0);
    std::vector<double> gradient_u_z(u.size(), 0.0);

    // Compute laplacian of u
    compute_laplacian(u, laplacian_u, dx, dy, dz, nx, ny, nz);

    // Compute gradient of u
    std::vector<double> dfdx(u.size(), 0.0);
    std::vector<double> dfdy(u.size(), 0.0);
    std::vector<double> dfdz(u.size(), 0.0);
    compute_gradient(u, dfdx, dfdy, dfdz, dx, dy, dz, nx, ny, nz);

    gradient_u_x = dfdx;
    gradient_u_y = dfdy;
    gradient_u_z = dfdz;

    // Compute divergence of u
    std::vector<double> divergence_u(u.size() / 3, 0.0);

    #pragma omp parallel for
    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        double div = gradient_u_x[idx] + gradient_u_y[idx + nx * ny * nz] + gradient_u_z[idx + 2 * nx * ny * nz];
        divergence_u[idx] = div;
    }

    // Compute gradient of divergence_u
    std::vector<double> gradient_divergence_u_x(u.size() / 3, 0.0);
    std::vector<double> gradient_divergence_u_y(u.size() / 3, 0.0);
    std::vector<double> gradient_divergence_u_z(u.size() / 3, 0.0);

    compute_gradient(divergence_u, gradient_divergence_u_x, gradient_divergence_u_y, gradient_divergence_u_z, dx, dy, dz, nx, ny, nz);

    // Compute convective term (u · ∇)u
    std::vector<double> convective_term(u.size(), 0.0);

    #pragma omp parallel for
    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        // For each component
        for (int comp = 0; comp < 3; ++comp) {
            int idx_comp = comp * nx * ny * nz + idx;
            double u_dot_grad_u = u[idx] * gradient_u_x[idx_comp] + u[idx + nx * ny * nz] * gradient_u_y[idx_comp] + u[idx + 2 * nx * ny * nz] * gradient_u_z[idx_comp];
            convective_term[idx_comp] = u_dot_grad_u;
        }
    }

    // Compute rate
    #pragma omp parallel for
    for (int idx = 0; idx < nx * ny * nz * 3; ++idx) {
        int idx_scalar = idx % (nx * ny * nz);
        double f_u = convective_term[idx]
            - shear_viscosity * laplacian_u[idx]
            - (bulk_viscosity + shear_viscosity / 3.0) * (idx < nx * ny * nz ? gradient_divergence_u_x[idx_scalar]
                                                                             : idx < 2 * nx * ny * nz ? gradient_divergence_u_y[idx_scalar]
                                                                                                      : gradient_divergence_u_z[idx_scalar]);

        rate[idx] = -f_u - 0.1 * vector_array[idx];

        // Apply boundary mask
        if (boundary_mask[idx_scalar]) {
            rate[idx] = 0.0;
        }
    }
}

// Adaptive Runge-Kutta method
void adaptive_runge_kutta(std::vector<double>& u, double& t, double& dt,
                          const std::vector<double>& vector_array,
                          const std::vector<bool>& boundary_mask,
                          int nx, int ny, int nz) {

    bool step_accepted = false;
    double dt_new = dt;

    int total_size = u.size();

    // Temporary variables
    std::vector<double> u_temp(total_size, 0.0);
    std::vector<double> k1(total_size, 0.0);
    std::vector<double> k2(total_size, 0.0);
    std::vector<double> k3(total_size, 0.0);
    std::vector<double> k4(total_size, 0.0);
    std::vector<double> k5(total_size, 0.0);

    while (!step_accepted) {
        // Compute k1
        evolution_rate(u, k1, vector_array, boundary_mask, nx, ny, nz);

        // Compute k2
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + 0.25 * dt * k1[idx];
        evolution_rate(u_temp, k2, vector_array, boundary_mask, nx, ny, nz);

        // Compute k3
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (3.0 / 32.0) * dt * k1[idx] + (9.0 / 32.0) * dt * k2[idx];
        evolution_rate(u_temp, k3, vector_array, boundary_mask, nx, ny, nz);

        // Compute k4
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (1932.0 / 2197.0) * dt * k1[idx]
                                  - (7200.0 / 2197.0) * dt * k2[idx]
                                  + (7296.0 / 2197.0) * dt * k3[idx];
        evolution_rate(u_temp, k4, vector_array, boundary_mask, nx, ny, nz);

        // Compute k5
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (439.0 / 216.0) * dt * k1[idx]
                                  - 8.0 * dt * k2[idx]
                                  + (3680.0 / 513.0) * dt * k3[idx]
                                  - (845.0 / 4104.0) * dt * k4[idx];
        evolution_rate(u_temp, k5, vector_array, boundary_mask, nx, ny, nz);

        // Compute 4th order solution
        std::vector<double> u4th(total_size, 0.0);
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u4th[idx] = u[idx] + dt * (
                (25.0 / 216.0) * k1[idx]
                + (1408.0 / 2565.0) * k3[idx]
                + (2197.0 / 4104.0) * k4[idx]
                - (1.0 / 5.0) * k5[idx]);

        // Compute 5th order solution
        std::vector<double> u5th(total_size, 0.0);
        #pragma omp parallel for
        for (int idx = 0; idx < total_size; ++idx)
            u5th[idx] = u[idx] + dt * (
                (16.0 / 135.0) * k1[idx]
                + (6656.0 / 12825.0) * k3[idx]
                + (28561.0 / 56430.0) * k4[idx]
                - (9.0 / 50.0) * k5[idx]
                + (2.0 / 55.0) * k5[idx]);

        // Estimate error
        double max_error = 0.0;
        #pragma omp parallel for reduction(max:max_error)
        for (int idx = 0; idx < total_size; ++idx) {
            double err = std::abs(u5th[idx] - u4th[idx]);
            double tol = tol_abs + tol_rel * std::max(std::abs(u4th[idx]), std::abs(u5th[idx]));
            double norm_err = err / tol;
            if (norm_err > max_error) {
                max_error = norm_err;
            }
        }

        // Decide if step is accepted
        if (max_error <= 1.0) {
            // Step is accepted
            step_accepted = true;
            u = u5th;
            t += dt;

            // Compute new dt
            dt_new = dt * std::min(dt_safety * std::pow(max_error, -0.2), 5.0);
            dt_new = std::min(dt_new, dt_max);
        } else {
            // Step is rejected, decrease dt
            dt_new = dt * std::max(dt_safety * std::pow(max_error, -0.25), 0.1);
            dt_new = std::max(dt_new, dt_min);
            // Ensure dt does not become too small
            if (dt_new < dt_min) {
                std::cerr << "Error: Time step below minimum allowed value." << std::endl;
                exit(1);
            }
            dt = dt_new;
        }
    }

    // Update dt for the next step
    dt = dt_new;
}

int main() {
    using namespace std;

    // Set the number of threads (optional)
    // omp_set_num_threads(4);  // Uncomment to set number of threads programmatically

    // Get the number of threads
    int num_threads = omp_get_max_threads();
    cout << "Number of threads: " << num_threads << endl;

    int total_size = nx * ny * nz;
    int vector_size = total_size * 3;  // For 3 components

    // Create grid
    vector<double> x = linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, nx);
    vector<double> y = linspace(Y_RANGE[0] + dy / 2, Y_RANGE[1] - dy / 2, ny);
    vector<double> z = linspace(Z_RANGE[0] + dz / 2, Z_RANGE[1] - dz / 2, nz);

    // Initialize grid matrices
    vector<double> x_grid(total_size, 0.0);
    vector<double> y_grid(total_size, 0.0);
    vector<double> z_grid(total_size, 0.0);

    #pragma omp parallel for
    for (int idx = 0; idx < total_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;
        x_grid[idx] = x[i];
        y_grid[idx] = y[j];
        z_grid[idx] = z[k];
    }

    // Initialize fields
    vector<double> u(vector_size, 0.0);
    vector<double> vector_array(vector_size, 0.0);

    int center = nx / 2;

    #pragma omp parallel for
    for (int idx = 0; idx < total_size; ++idx) {
        int i = idx / (ny * nz);
        if (i == center - 1 || i == center) {
            vector_array[idx] = 1.0;  // Component 0
        }
    }

    // Define boundary mask
    double y_width = 0.5;
    int y_count = 5;
    double z_width = 0.5;
    int z_count = 5;

    vector<bool> boundary_mask(total_size, false);

    #pragma omp parallel for
    for (int idx = 0; idx < total_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;

        bool in_x_range = (x_grid[idx] >= 3.0) && (x_grid[idx] <= 7.0);
        double y_mod = fmod(y_grid[idx], 5.0 / y_count);
        double z_mod = fmod(z_grid[idx], 5.0 / z_count);
        bool y_condition = (y_mod >= (5.0 / y_count - y_width / 2)) || (y_mod <= y_width / 2);
        bool z_condition = (z_mod >= (5.0 / z_count - z_width / 2)) || (z_mod <= z_width / 2);
        if (in_x_range && (y_condition || z_condition)) {
            boundary_mask[idx] = true;
        }
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Time-stepping loop
    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        adaptive_runge_kutta(u, t, dt, vector_array, boundary_mask, nx, ny, nz);
        step++;
        if (step % 10 == 0) {
            std::cout << "Time: " << t << " / " << t_end << ", dt: " << dt << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;
    std::cout << "Execution Time: " << execution_time.count() << " seconds" << std::endl;

    // Write the vector field to a csv file
    ofstream file("output.csv");
    for (int idx = 0; idx < total_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;
        file << x_grid[idx] << "," << y_grid[idx] << "," << z_grid[idx] << ","
             << u[idx] << "," << u[idx + total_size] << "," << u[idx + 2 * total_size] << endl;
    }
    file.close();

    return 0;
}
