#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>


// Grid parameters
const int X_SIZE = 160;  // Should be divisible by 2
const int Y_SIZE = 80;
const int Z_SIZE = 80;
const double X_RANGE[2] = {0.0, 10.0};
const double Y_RANGE[2] = {0.0, 5.0};
const double Z_RANGE[2] = {0.0, 5.0};

const double dx = (X_RANGE[1] - X_RANGE[0]) / X_SIZE;
const double dy = (Y_RANGE[1] - Y_RANGE[0]) / Y_SIZE;
const double dz = (Z_RANGE[1] - Z_RANGE[0]) / Z_SIZE;

// Time parameters
const double t_range = 1;
const double dt = 0.001;
const int num_steps = static_cast<int>(t_range / dt);

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

// Template function to create a 3D vector
template<typename T>
std::vector<std::vector<std::vector<T>>> create_3D_vector(int dim1, int dim2, int dim3, T init_value) {
    return std::vector<std::vector<std::vector<T>>>(dim1,
        std::vector<std::vector<T>>(dim2,
            std::vector<T>(dim3, init_value)));
}

// Template function to create a 4D vector
template<typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> create_4D_vector(int dim1, int dim2, int dim3, int dim4, T init_value) {
    return std::vector<std::vector<std::vector<std::vector<T>>>>(dim1,
        std::vector<std::vector<std::vector<T>>>(dim2,
            std::vector<std::vector<T>>(dim3,
                std::vector<T>(dim4, init_value))));
}

// Template function to create a 5D vector
template<typename T>
std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>> create_5D_vector(int dim1, int dim2, int dim3, int dim4, int dim5, T init_value) {
    return std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>(dim1,
        std::vector<std::vector<std::vector<std::vector<T>>>>(dim2,
            std::vector<std::vector<std::vector<T>>>(dim3,
                std::vector<std::vector<T>>(dim4,
                    std::vector<T>(dim5, init_value)))));
}

// Helper functions for finite differences
void compute_gradient(const std::vector<std::vector<std::vector<double>>>& f,
                      std::vector<std::vector<std::vector<double>>>& dfdx,
                      std::vector<std::vector<std::vector<double>>>& dfdy,
                      std::vector<std::vector<std::vector<double>>>& dfdz,
                      double dx, double dy, double dz) {
    int nx = f.size();
    int ny = f[0].size();
    int nz = f[0][0].size();

    // Compute dfdx
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                dfdx[i][j][k] = (f[i + 1][j][k] - f[i - 1][j][k]) / (2 * dx);
            }
        }
    }
    // Boundary conditions for dfdx (Dirichlet: f=0)
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            dfdx[0][j][k] = (f[1][j][k] - 0) / dx;
            dfdx[nx - 1][j][k] = (0 - f[nx - 2][j][k]) / dx;
        }
    }

    // Compute dfdy
    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 0; k < nz; ++k) {
                dfdy[i][j][k] = (f[i][j + 1][k] - f[i][j - 1][k]) / (2 * dy);
            }
        }
    }
    // Boundary conditions for dfdy (Dirichlet: f=0)
    for (int i = 0; i < nx; ++i) {
        for (int k = 0; k < nz; ++k) {
            dfdy[i][0][k] = (f[i][1][k] - 0) / dy;
            dfdy[i][ny - 1][k] = (0 - f[i][ny - 2][k]) / dy;
        }
    }

    // Compute dfdz
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                dfdz[i][j][k] = (f[i][j][k + 1] - f[i][j][k - 1]) / (2 * dz);
            }
        }
    }
    // Boundary conditions for dfdz (Dirichlet: f=0)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            dfdz[i][j][0] = (f[i][j][1] - 0) / dz;
            dfdz[i][j][nz - 1] = (0 - f[i][j][nz - 2]) / dz;
        }
    }
}

void compute_laplacian(const std::vector<std::vector<std::vector<double>>>& f,
                       std::vector<std::vector<std::vector<double>>>& laplacian_f,
                       double dx, double dy, double dz) {
    int nx = f.size();
    int ny = f[0].size();
    int nz = f[0][0].size();

    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                double d2fdx2 = (f[i + 1][j][k] - 2 * f[i][j][k] + f[i - 1][j][k]) / (dx * dx);
                double d2fdy2 = (f[i][j + 1][k] - 2 * f[i][j][k] + f[i][j - 1][k]) / (dy * dy);
                double d2fdz2 = (f[i][j][k + 1] - 2 * f[i][j][k] + f[i][j][k - 1]) / (dz * dz);
                laplacian_f[i][j][k] = d2fdx2 + d2fdy2 + d2fdz2;
            }
        }
    }

    // Boundary conditions for laplacian (Dirichlet: f=0)
    // Set laplacian to zero at boundaries
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            laplacian_f[i][j][0] = 0.0;
            laplacian_f[i][j][nz - 1] = 0.0;
        }
        for (int k = 0; k < nz; ++k) {
            laplacian_f[i][0][k] = 0.0;
            laplacian_f[i][ny - 1][k] = 0.0;
        }
    }
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            laplacian_f[0][j][k] = 0.0;
            laplacian_f[nx - 1][j][k] = 0.0;
        }
    }
}

// Evolution rate function
void evolution_rate(const std::vector<std::vector<std::vector<std::vector<double>>>>& u,
                    std::vector<std::vector<std::vector<std::vector<double>>>>& rate,
                    const std::vector<std::vector<std::vector<std::vector<double>>>>& vector_array,
                    const std::vector<std::vector<std::vector<bool>>>& boundary_mask) {
    int nx = u[0].size();
    int ny = u[0][0].size();
    int nz = u[0][0][0].size();

    // Initialize intermediate variables
    auto laplacian_u = create_4D_vector<double>(3, nx, ny, nz, 0.0);
    auto gradient_u = create_5D_vector<double>(3, 3, nx, ny, nz, 0.0);
    auto divergence_u = create_3D_vector<double>(nx, ny, nz, 0.0);
    auto gradient_divergence_u = create_4D_vector<double>(3, nx, ny, nz, 0.0);

    // Compute laplacian of u
    for (int comp = 0; comp < 3; ++comp) {
        compute_laplacian(u[comp], laplacian_u[comp], dx, dy, dz);
    }

    // Compute gradient of u
    for (int comp = 0; comp < 3; ++comp) {
        std::vector<std::vector<std::vector<double>>> dfdx(nx,
            std::vector<std::vector<double>>(ny,
            std::vector<double>(nz, 0.0)));
        std::vector<std::vector<std::vector<double>>> dfdy = dfdx;
        std::vector<std::vector<std::vector<double>>> dfdz = dfdx;
        compute_gradient(u[comp], dfdx, dfdy, dfdz, dx, dy, dz);
        gradient_u[comp][0] = dfdx;
        gradient_u[comp][1] = dfdy;
        gradient_u[comp][2] = dfdz;
    }

    // Compute divergence of u
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                divergence_u[i][j][k] = 0.0;
                for (int comp = 0; comp < 3; ++comp) {
                    double dudx = gradient_u[comp][0][i][j][k];
                    double dudy = gradient_u[comp][1][i][j][k];
                    double dudz = gradient_u[comp][2][i][j][k];
                    if (comp == 0) {
                        divergence_u[i][j][k] += dudx;
                    } else if (comp == 1) {
                        divergence_u[i][j][k] += dudy;
                    } else {
                        divergence_u[i][j][k] += dudz;
                    }
                }
            }
        }
    }

    // Compute gradient of divergence_u
    std::vector<std::vector<std::vector<double>>> ddivdx(nx,
        std::vector<std::vector<double>>(ny,
        std::vector<double>(nz, 0.0)));
    std::vector<std::vector<std::vector<double>>> ddivdy = ddivdx;
    std::vector<std::vector<std::vector<double>>> ddivdz = ddivdx;
    compute_gradient(divergence_u, ddivdx, ddivdy, ddivdz, dx, dy, dz);
    gradient_divergence_u[0] = ddivdx;
    gradient_divergence_u[1] = ddivdy;
    gradient_divergence_u[2] = ddivdz;

    // Compute convective term (u · ∇)u
    auto convective_term = create_4D_vector<double>(3, nx, ny, nz, 0.0);

    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    double dot_product = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        dot_product += u[d][i][j][k] * gradient_u[comp][d][i][j][k];
                    }
                    convective_term[comp][i][j][k] = dot_product;
                }
            }
        }
    }

    // Compute f_u
    auto f_u = create_4D_vector<double>(3, nx, ny, nz, 0.0);

    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    f_u[comp][i][j][k] = convective_term[comp][i][j][k]
                        - shear_viscosity * laplacian_u[comp][i][j][k]
                        - (bulk_viscosity + shear_viscosity / 3.0) * gradient_divergence_u[comp][i][j][k];
                }
            }
        }
    }

    // Compute rate
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    rate[comp][i][j][k] = -f_u[comp][i][j][k] - 0.1 * vector_array[comp][i][j][k];
                    // Apply boundary mask
                    if (boundary_mask[i][j][k]) {
                        rate[comp][i][j][k] = 0.0;
                    }
                }
            }
        }
    }
}

// Runge-Kutta 4 method
void runge_kutta_4(std::vector<std::vector<std::vector<std::vector<double>>>>& u, double t, double dt,
                   const std::vector<std::vector<std::vector<std::vector<double>>>>& vector_array,
                   const std::vector<std::vector<std::vector<bool>>>& boundary_mask) {
    int nx = u[0].size();
    int ny = u[0][0].size();
    int nz = u[0][0][0].size();

    auto u_temp = u;
    auto k1 = create_4D_vector<double>(3, nx, ny, nz, 0.0);
    auto k2 = create_4D_vector<double>(3, nx, ny, nz, 0.0);
    auto k3 = create_4D_vector<double>(3, nx, ny, nz, 0.0);
    auto k4 = create_4D_vector<double>(3, nx, ny, nz, 0.0);

    evolution_rate(u, k1, vector_array, boundary_mask);

    // u_temp = u + 0.5 * dt * k1
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    u_temp[comp][i][j][k] = u[comp][i][j][k] + 0.5 * dt * k1[comp][i][j][k];
                }
            }
        }
    }

    evolution_rate(u_temp, k2, vector_array, boundary_mask);

    // u_temp = u + 0.5 * dt * k2
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    u_temp[comp][i][j][k] = u[comp][i][j][k] + 0.5 * dt * k2[comp][i][j][k];
                }
            }
        }
    }

    evolution_rate(u_temp, k3, vector_array, boundary_mask);

    // u_temp = u + dt * k3
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    u_temp[comp][i][j][k] = u[comp][i][j][k] + dt * k3[comp][i][j][k];
                }
            }
        }
    }

    evolution_rate(u_temp, k4, vector_array, boundary_mask);

    // u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    u[comp][i][j][k] += (dt / 6.0) * (k1[comp][i][j][k] + 2 * k2[comp][i][j][k] + 2 * k3[comp][i][j][k] + k4[comp][i][j][k]);
                }
            }
        }
    }
}



int main() {
    using namespace std::chrono;

    // Create grid
    std::vector<double> x = linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, X_SIZE);
    std::vector<double> y = linspace(Y_RANGE[0] + dy / 2, Y_RANGE[1] - dy / 2, Y_SIZE);
    std::vector<double> z = linspace(Z_RANGE[0] + dz / 2, Z_RANGE[1] - dz / 2, Z_SIZE);

    // Initialize grid matrices
    auto x_grid = create_3D_vector<double>(X_SIZE, Y_SIZE, Z_SIZE, 0.0);
    auto y_grid = x_grid;
    auto z_grid = x_grid;

    for (int i = 0; i < X_SIZE; ++i) {
        for (int j = 0; j < Y_SIZE; ++j) {
            for (int k = 0; k < Z_SIZE; ++k) {
                x_grid[i][j][k] = x[i];
                y_grid[i][j][k] = y[j];
                z_grid[i][j][k] = z[k];
            }
        }
    }

    // Initialize fields
    auto u = create_4D_vector<double>(3, X_SIZE, Y_SIZE, Z_SIZE, 0.0);
    auto vector_array = u;

    int center = X_SIZE / 2;
    for (int i = center - 1; i <= center; ++i) {
        for (int j = 0; j < Y_SIZE; ++j) {
            for (int k = 0; k < Z_SIZE; ++k) {
                vector_array[0][i][j][k] = 1.0;
            }
        }
    }

    // Define boundary mask
    double y_width = 0.5;
    int y_count = 5;
    double z_width = 0.5;
    int z_count = 5;

    auto boundary_mask = create_3D_vector<bool>(X_SIZE, Y_SIZE, Z_SIZE, false);

    for (int i = 0; i < X_SIZE; ++i) {
        for (int j = 0; j < Y_SIZE; ++j) {
            for (int k = 0; k < Z_SIZE; ++k) {
                bool in_x_range = (x_grid[i][j][k] >= 3.0) && (x_grid[i][j][k] <= 7.0);
                double y_mod = fmod(y_grid[i][j][k], 5.0 / y_count);
                double z_mod = fmod(z_grid[i][j][k], 5.0 / z_count);
                bool y_condition = (y_mod >= (5.0 / y_count - y_width / 2)) || (y_mod <= y_width / 2);
                bool z_condition = (z_mod >= (5.0 / z_count - z_width / 2)) || (z_mod <= z_width / 2);
                if (in_x_range && (y_condition || z_condition)) {
                    boundary_mask[i][j][k] = true;
                }
            }
        }
    }

    // Start timing
    auto start_time = high_resolution_clock::now();

    // Time-stepping loop
    double t = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        runge_kutta_4(u, t, dt, vector_array, boundary_mask);

        t += dt;
        if (step % 10 == 0) {
            std::cout << "Time: " << t << " / " << t_range << std::endl;
        }
    }

    auto end_time = high_resolution_clock::now();
    duration<double> execution_time = end_time - start_time;
    std::cout << "Execution Time: " << execution_time.count() << " seconds" << std::endl;


    // Store results in .csv file
    std::ofstream file("output.csv");
    for (int i = 0; i < X_SIZE; ++i) {
        for (int j = 0; j < Y_SIZE; ++j) {
            for (int k = 0; k < Z_SIZE; ++k) {
                file << u[0][i][j][k] << "," << u[1][i][j][k] << "," << u[2][i][j][k] << std::endl;
            }
        }
    }
    file.close();

    return 0;
}
