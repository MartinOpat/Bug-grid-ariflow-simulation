// pde_solver_openmp_fixed.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <fstream>

// Grid parameters
const int X_SIZE = 160;  // Adjust as needed
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

// Pre-allocated global arrays to reduce stack usage
std::vector<double> laplacian_u;
std::vector<double> gradient_u_x;
std::vector<double> gradient_u_y;
std::vector<double> gradient_u_z;
std::vector<double> divergence_u;
std::vector<double> grad_div_x;
std::vector<double> grad_div_y;
std::vector<double> grad_div_z;

// Evolution rate function
void evolution_rate(const std::vector<double>& u,
                    std::vector<double>& rate,
                    const std::vector<double>& vector_array,
                    const std::vector<bool>& boundary_mask,
                    int nx, int ny, int nz) {

    int scalar_size = nx * ny * nz;
    int total_size = scalar_size * 3;  // 3 components

    // Ensure the global arrays are correctly sized
    laplacian_u.resize(total_size);
    gradient_u_x.resize(total_size);
    gradient_u_y.resize(total_size);
    gradient_u_z.resize(total_size);
    divergence_u.resize(scalar_size);
    grad_div_x.resize(scalar_size);
    grad_div_y.resize(scalar_size);
    grad_div_z.resize(scalar_size);

    // Compute gradient and laplacian for each component
    for (int comp = 0; comp < 3; ++comp) {
        int offset = comp * scalar_size;

        // Compute gradients and laplacian
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < scalar_size; ++idx) {
            int i = idx / (ny * nz);
            int j = (idx / nz) % ny;
            int k = idx % nz;

            int i_p = (i < nx - 1) ? i + 1 : i;  // Neumann boundary
            int i_m = (i > 0) ? i - 1 : i;       // Neumann boundary
            int j_p = (j < ny - 1) ? j + 1 : j;
            int j_m = (j > 0) ? j - 1 : j;
            int k_p = (k < nz - 1) ? k + 1 : k;
            int k_m = (k > 0) ? k - 1 : k;

            int idx_ip = index(i_p, j, k, ny, nz);
            int idx_im = index(i_m, j, k, ny, nz);
            int idx_jp = index(i, j_p, k, ny, nz);
            int idx_jm = index(i, j_m, k, ny, nz);
            int idx_kp = index(i, j, k_p, ny, nz);
            int idx_km = index(i, j, k_m, ny, nz);

            // Gradient
            gradient_u_x[offset + idx] = (u[offset + idx_ip] - u[offset + idx_im]) / (2 * dx);
            gradient_u_y[offset + idx] = (u[offset + idx_jp] - u[offset + idx_jm]) / (2 * dy);
            gradient_u_z[offset + idx] = (u[offset + idx_kp] - u[offset + idx_km]) / (2 * dz);

            // Laplacian
            double d2fdx2 = (u[offset + idx_ip] - 2 * u[offset + idx] + u[offset + idx_im]) / (dx * dx);
            double d2fdy2 = (u[offset + idx_jp] - 2 * u[offset + idx] + u[offset + idx_jm]) / (dy * dy);
            double d2fdz2 = (u[offset + idx_kp] - 2 * u[offset + idx] + u[offset + idx_km]) / (dz * dz);
            laplacian_u[offset + idx] = d2fdx2 + d2fdy2 + d2fdz2;
        }
    }

    // Compute divergence of u
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
        divergence_u[idx] = gradient_u_x[0 * scalar_size + idx] + gradient_u_y[1 * scalar_size + idx] + gradient_u_z[2 * scalar_size + idx];
    }

    // Compute gradient of divergence_u
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;

        int i_p = (i < nx - 1) ? i + 1 : i;
        int i_m = (i > 0) ? i - 1 : i;
        int j_p = (j < ny - 1) ? j + 1 : j;
        int j_m = (j > 0) ? j - 1 : j;
        int k_p = (k < nz - 1) ? k + 1 : k;
        int k_m = (k > 0) ? k - 1 : k;

        int idx_ip = index(i_p, j, k, ny, nz);
        int idx_im = index(i_m, j, k, ny, nz);
        int idx_jp = index(i, j_p, k, ny, nz);
        int idx_jm = index(i, j_m, k, ny, nz);
        int idx_kp = index(i, j, k_p, ny, nz);
        int idx_km = index(i, j, k_m, ny, nz);

        grad_div_x[idx] = (divergence_u[idx_ip] - divergence_u[idx_im]) / (2 * dx);
        grad_div_y[idx] = (divergence_u[idx_jp] - divergence_u[idx_jm]) / (2 * dy);
        grad_div_z[idx] = (divergence_u[idx_kp] - divergence_u[idx_km]) / (2 * dz);
    }

    // Compute convective term and rate
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
        for (int comp = 0; comp < 3; ++comp) {
            int idx_comp = comp * scalar_size + idx;

            double u_dot_grad_u = u[0 * scalar_size + idx] * gradient_u_x[idx_comp]
                                + u[1 * scalar_size + idx] * gradient_u_y[idx_comp]
                                + u[2 * scalar_size + idx] * gradient_u_z[idx_comp];

            double grad_div = (comp == 0) ? grad_div_x[idx] : (comp == 1) ? grad_div_y[idx] : grad_div_z[idx];

            double f_u = u_dot_grad_u
                - shear_viscosity * laplacian_u[idx_comp]
                - (bulk_viscosity + shear_viscosity / 3.0) * grad_div;

            rate[idx_comp] = -f_u - 0.1 * vector_array[idx_comp];

            // Apply boundary mask
            if (boundary_mask[idx]) {
                rate[idx_comp] = 0.0;
            }
        }
    }
}

// Pre-allocated global arrays for adaptive Runge-Kutta
std::vector<double> u_temp;
std::vector<double> k1, k2, k3, k4, k5;
std::vector<double> u4th, u5th;

// Adaptive Runge-Kutta method
void adaptive_runge_kutta(std::vector<double>& u, double& t, double& dt,
                          const std::vector<double>& vector_array,
                          const std::vector<bool>& boundary_mask,
                          int nx, int ny, int nz) {

    int total_size = nx * ny * nz * 3;  // 3 components

    // Ensure global arrays are correctly sized
    u_temp.resize(total_size);
    k1.resize(total_size);
    k2.resize(total_size);
    k3.resize(total_size);
    k4.resize(total_size);
    k5.resize(total_size);
    u4th.resize(total_size);
    u5th.resize(total_size);

    bool step_accepted = false;
    double dt_new = dt;

    while (!step_accepted) {
        // Compute k1
        evolution_rate(u, k1, vector_array, boundary_mask, nx, ny, nz);

        // Compute k2
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + 0.25 * dt * k1[idx];
        evolution_rate(u_temp, k2, vector_array, boundary_mask, nx, ny, nz);

        // Compute k3
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (3.0 / 32.0) * dt * k1[idx] + (9.0 / 32.0) * dt * k2[idx];
        evolution_rate(u_temp, k3, vector_array, boundary_mask, nx, ny, nz);

        // Compute k4
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (1932.0 / 2197.0) * dt * k1[idx]
                                  - (7200.0 / 2197.0) * dt * k2[idx]
                                  + (7296.0 / 2197.0) * dt * k3[idx];
        evolution_rate(u_temp, k4, vector_array, boundary_mask, nx, ny, nz);

        // Compute k5
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_size; ++idx)
            u_temp[idx] = u[idx] + (439.0 / 216.0) * dt * k1[idx]
                                  - 8.0 * dt * k2[idx]
                                  + (3680.0 / 513.0) * dt * k3[idx]
                                  - (845.0 / 4104.0) * dt * k4[idx];
        evolution_rate(u_temp, k5, vector_array, boundary_mask, nx, ny, nz);

        // Compute 4th order solution
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_size; ++idx)
            u4th[idx] = u[idx] + dt * (
                (25.0 / 216.0) * k1[idx]
                + (1408.0 / 2565.0) * k3[idx]
                + (2197.0 / 4104.0) * k4[idx]
                - (1.0 / 5.0) * k5[idx]);

        // Compute 5th order solution
        #pragma omp parallel for schedule(static)
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
        }

        dt = dt_new;
    }
}

// Main function
int main(int argc, char* argv[]) {

    // Increase OpenMP stack size
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    // omp_set_num_threads(omp_get_max_threads());
    // omp_set_stacksize(512 * 1024 * 1024); // 512 MB

    using namespace std;

    int nx = X_SIZE;
    int ny = Y_SIZE;
    int nz = Z_SIZE;

    int scalar_size = nx * ny * nz;
    int total_size = scalar_size * 3;  // For 3 components

    vector<double> x = linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, nx);
    vector<double> y = linspace(Y_RANGE[0] + dy / 2, Y_RANGE[1] - dy / 2, ny);
    vector<double> z = linspace(Z_RANGE[0] + dz / 2, Z_RANGE[1] - dz / 2, nz);

    // Initialize grid matrices
    vector<double> x_grid(scalar_size, 0.0);
    vector<double> y_grid(scalar_size, 0.0);
    vector<double> z_grid(scalar_size, 0.0);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;

        x_grid[idx] = x[i];
        y_grid[idx] = y[j];
        z_grid[idx] = z[k];
    }

    // Initialize fields
    vector<double> u(total_size, 0.0);
    vector<double> vector_array(total_size, 0.0);

    int center = nx / 2;

    // Initialize vector_array
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
        int i = idx / (ny * nz);
        if (i == center - 1 || i == center) {
            for (int comp = 0; comp < 3; ++comp) {
                vector_array[comp * scalar_size + idx] = 1.0;
            }
        }
    }

    // Define boundary mask
    double y_width = 0.5;
    int y_count = 5;
    double z_width = 0.5;
    int z_count = 5;

    vector<bool> boundary_mask(scalar_size, false);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < scalar_size; ++idx) {
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

    // Store the results in a .csv file
    ofstream output_file("results.csv");
    for (int idx = 0; idx < total_size; ++idx) {
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;
        output_file << x_grid[idx] << "," << y_grid[idx] << "," << z_grid[idx] << ","
                    << u[0 * scalar_size + idx] << "," << u[1 * scalar_size + idx] << "," << u[2 * scalar_size + idx] << std::endl;
    }
    output_file.close();

    return 0;
}
