#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <fstream>

#define X_SIZE 100 // should be divisible by 2
#define Y_SIZE 50
#define Z_SIZE 50

#define DX 1.0
#define DY 1.0
#define DZ 1.0

using namespace Eigen;
using namespace std;

// Define a vector field of size X_SIZE x Y_SIZE x Z_SIZE
struct VectorField {
    Array<double, Dynamic, Dynamic> data;

    VectorField() : data(3, X_SIZE * Y_SIZE * Z_SIZE) {
        data.setZero();
        // Setting initial condition
        for (int x = X_SIZE / 2 - 1; x <= X_SIZE / 2; x++) {
            for (int y = 0; y < Y_SIZE; y++) {
                for (int z = 0; z < Z_SIZE; z++) {
                    int index = (z * Y_SIZE + y) * X_SIZE + x;
                    data(0, index) = 1;  // Set initial condition for X component
                }
            }
        }
    }

    // Non-const version for modification
    double& operator()(int component, int x, int y, int z) {
        int index = (z * Y_SIZE + y) * X_SIZE + x;
        return data(component, index);
    }

    // Const version for read-only access
    double operator()(int component, int x, int y, int z) const {
        int index = (z * Y_SIZE + y) * X_SIZE + x;
        return data(component, index);
    }
};

struct ScalarField {
    Array<double, Dynamic, 1> data;

    ScalarField(double initialValue) : data(X_SIZE * Y_SIZE * Z_SIZE) {
        data.setConstant(initialValue);
    }

    // Non-const version for modification
    double& operator()(int x, int y, int z) {
        int index = (z * Y_SIZE + y) * X_SIZE + x;
        return data(index);
    }

    // Const version for read-only access
    double operator()(int x, int y, int z) const {
        int index = (z * Y_SIZE + y) * X_SIZE + x;
        return data(index);
    }
};

// Custom boundary mask computation
vector<vector<vector<bool>>> computeBoundaryMask() {
    vector<vector<vector<bool>>> boundaryMask(X_SIZE, vector<vector<bool>>(Y_SIZE, vector<bool>(Z_SIZE, false)));
    double y_width = 0.2;
    int y_count = 6;
    double z_width = 0.2;
    int z_count = 6;

    for (int x = 0; x < X_SIZE; ++x) {
        for (int y = 0; y < Y_SIZE; ++y) {
            for (int z = 0; z < Z_SIZE; ++z) {
                double y_mod = fmod(y, 5.0 / y_count);
                double z_mod = fmod(z, 5.0 / z_count);
                bool y_condition = (y_mod >= 5.0 / y_count - y_width / 2) || (y_mod <= y_width / 2);
                bool z_condition = (z_mod >= 5.0 / z_count - z_width / 2) || (z_mod <= z_width / 2);
                if ((3 <= x && x <= 7) && (y_condition || z_condition)) {
                    boundaryMask[x][y][z] = true;
                }
            }
        }
    }
    return boundaryMask;
}

// Compute gradient of a scalar field
VectorField gradient(const ScalarField& field) {
    VectorField grad;

    for (int z = 1; z < Z_SIZE - 1; z++) {
        for (int y = 1; y < Y_SIZE - 1; y++) {
            for (int x = 1; x < X_SIZE - 1; x++) {
                grad(0, x, y, z) = (field(x + 1, y, z) - field(x - 1, y, z)) / (2 * DX);
                grad(1, x, y, z) = (field(x, y + 1, z) - field(x, y - 1, z)) / (2 * DY);
                grad(2, x, y, z) = (field(x, y, z + 1) - field(x, y, z - 1)) / (2 * DZ);
            }
        }
    }

    return grad;
}

// Compute divergence of a vector field
ScalarField divergence(const VectorField& field) {
    ScalarField div(0.0);

    for (int z = 1; z < Z_SIZE - 1; z++) {
        for (int y = 1; y < Y_SIZE - 1; y++) {
            for (int x = 1; x < X_SIZE - 1; x++) {
                double d_dx = (field(0, x + 1, y, z) - field(0, x - 1, y, z)) / (2 * DX);
                double d_dy = (field(1, x, y + 1, z) - field(1, x, y - 1, z)) / (2 * DY);
                double d_dz = (field(2, x, y, z + 1) - field(2, x, y, z - 1)) / (2 * DZ);
                div(x, y, z) = d_dx + d_dy + d_dz;
            }
        }
    }

    return div;
}

// Compute Laplacian of a scalar field
ScalarField laplace(const ScalarField& field) {
    ScalarField lap(0.0);

    for (int z = 1; z < Z_SIZE - 1; z++) {
        for (int y = 1; y < Y_SIZE - 1; y++) {
            for (int x = 1; x < X_SIZE - 1; x++) {
                double d2_dx2 = (field(x + 1, y, z) - 2 * field(x, y, z) + field(x - 1, y, z)) / (DX * DX);
                double d2_dy2 = (field(x, y + 1, z) - 2 * field(x, y, z) + field(x, y - 1, z)) / (DY * DY);
                double d2_dz2 = (field(x, y, z + 1) - 2 * field(x, y, z) + field(x, y, z - 1)) / (DZ * DZ);
                lap(x, y, z) = d2_dx2 + d2_dy2 + d2_dz2;
            }
        }
    }

    return lap;
}

// Time evolution of the PDE
void evolutionRate(VectorField &state, ScalarField &density, VectorField &vectorArray, 
                   vector<vector<vector<bool>>> &boundaryMask, double dynamicViscosity, double bulkViscosity) {
    // Placeholder for actual implementation of PDE evolution step
    ScalarField density_laplace = laplace(density);

    // Loop through all grid points and compute the evolution rate
    for (int z = 1; z < Z_SIZE - 1; z++) {
        for (int y = 1; y < Y_SIZE - 1; y++) {
            for (int x = 1; x < X_SIZE - 1; x++) {
                if (boundaryMask[x][y][z]) {
                    state(0, x, y, z) = 0;
                    state(1, x, y, z) = 0;
                    state(2, x, y, z) = 0;
                } else {
                    // Example evolution logic involving viscosity
                    state(0, x, y, z) += dynamicViscosity * density_laplace(x, y, z);
                    state(1, x, y, z) += bulkViscosity * density_laplace(x, y, z);
                    state(2, x, y, z) += dynamicViscosity * density_laplace(x, y, z);
                }
            }
        }
    }
}

// Solving the PDE
void solvePDE(VectorField &field, ScalarField &density, VectorField &vectorArray, 
              vector<vector<vector<bool>>> &boundaryMask, double tRange, double dt) {
    double time = 0;
    double dynamicViscosity = 1.0;
    double bulkViscosity = 1.0;

    // Time loop
    while (time < tRange) {
        // print progres at every 5 percent
        if (fmod(time, tRange / 20) < dt) {
            cout << "Progress: " << time / tRange * 100 << "%" << endl;
        }
        evolutionRate(field, density, vectorArray, boundaryMask, dynamicViscosity, bulkViscosity);
        time += dt;
    }
}

int main() {
    // Initialize fields
    VectorField field;
    ScalarField density(10.0);
    VectorField vectorArray;

    // Compute boundary mask
    auto boundaryMask = computeBoundaryMask();

    // Start timer
    auto start = chrono::high_resolution_clock::now();

    // Solve PDE
    solvePDE(field, density, vectorArray, boundaryMask, 1.0, 0.01);

    // End timer
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Execution Time: " << elapsed.count() << " seconds" << endl;

    // Export the data to a .csv file
    ofstream file("output.csv");
    for (int z = 0; z < Z_SIZE; z++) {
        for (int y = 0; y < Y_SIZE; y++) {
            for (int x = 0; x < X_SIZE; x++) {
                file << field(0, x, y, z) << "," << field(1, x, y, z) << "," << field(2, x, y, z) << endl;
            }
        }
    }
    file.close();

    return 0;
}
