#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>

__constant__ double G = 6.674e-11; // constant accessible on GPU

struct simulation {
    size_t nbpart;

    std::vector<double> mass;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;

    simulation(size_t nb)
        : nbpart(nb), mass(nb), x(nb), y(nb), z(nb),
          vx(nb), vy(nb), vz(nb), fx(nb), fy(nb), fz(nb) {}
};

// CPU initialization
void random_init(simulation& s) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dismass(0.9, 1.0);
    std::normal_distribution<double> dispos(0., 1.);
    std::normal_distribution<double> disvel(0., 1.);

    for (size_t i = 0; i < s.nbpart; ++i) {
        s.mass[i] = dismass(gen);
        s.x[i] = dispos(gen);
        s.y[i] = dispos(gen);
        s.z[i] = 0.;
        s.vx[i] = s.y[i] * 1.5;
        s.vy[i] = -s.x[i] * 1.5;
        s.vz[i] = 0.;
    }
}

void init_solar(simulation& s) {
    enum Planets { SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON };
    s = simulation(10);

    s.mass = {
        1.9891e30, 3.285e23, 4.867e24, 5.972e24, 6.39e23,
        1.898e27, 5.683e26, 8.681e25, 1.024e26, 7.342e22
    };

    double AU = 1.496e11;
    s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844e8};
    s.y = std::vector<double>(10, 0);
    s.z = std::vector<double>(10, 0);
    s.vx = std::vector<double>(10, 0);
    s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
    s.vz = std::vector<double>(10, 0);
}

// CUDA kernels
__global__ void compute_forces(size_t n, double* mass, double* x, double* y, double* z,
                               double* fx, double* fy, double* fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
    double softening = 0.1;

    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;

        double dx = x[i] - x[j];
        double dy = y[i] - y[j];
        double dz = z[i] - z[j];
        double dist_sqr = dx*dx + dy*dy + dz*dz + softening;
        double inv_dist = rsqrt(dist_sqr);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;
        double F = G * mass[i] * mass[j] * inv_dist3;

        fx_i += dx * F;
        fy_i += dy * F;
        fz_i += dz * F;
    }

    fx[i] = fx_i;
    fy[i] = fy_i;
    fz[i] = fz_i;
}

__global__ void update_positions(size_t n, double dt, double* mass, double* x, double* y, double* z,
                                 double* vx, double* vy, double* vz,
                                 double* fx, double* fy, double* fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += fx[i] / mass[i] * dt;
    vy[i] += fy[i] / mass[i] * dt;
    vz[i] += fz[i] / mass[i] * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

// Utility to print state
void dump_state(size_t n, double* mass, double* x, double* y, double* z,
                double* vx, double* vy, double* vz, double* fx, double* fy, double* fz) {
    std::cout << n << '\t';
    for (size_t i = 0; i < n; ++i) {
        std::cout << mass[i] << '\t' << x[i] << '\t' << y[i] << '\t' << z[i] << '\t';
        std::cout << vx[i] << '\t' << vy[i] << '\t' << vz[i] << '\t';
        std::cout << fx[i] << '\t' << fy[i] << '\t' << fz[i] << '\t';
    }
    std::cout << '\n';
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <input> <dt> <nbstep> <printevery>\n"
                  << "input can be:\na number (random initialization)\nplanet\nfilename\n";
        return -1;
    }

    double dt = std::atof(argv[2]);
    size_t nbstep = std::atol(argv[3]);
    size_t printevery = std::atol(argv[4]);

    simulation s(1);

    size_t nbpart = std::atol(argv[1]);
    if (nbpart > 0) {
        s = simulation(nbpart);
        random_init(s);
    } else {
        std::string inputparam = argv[1];
        if (inputparam == "planet") init_solar(s);
        else throw std::runtime_error("File loading not implemented in CUDA version");
    }

    size_t n = s.nbpart;

    // Allocate device memory
    double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_mass, n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_z, n * sizeof(double));
    cudaMalloc(&d_vx, n * sizeof(double));
    cudaMalloc(&d_vy, n * sizeof(double));
    cudaMalloc(&d_vz, n * sizeof(double));
    cudaMalloc(&d_fx, n * sizeof(double));
    cudaMalloc(&d_fy, n * sizeof(double));
    cudaMalloc(&d_fz, n * sizeof(double));

    // Copy to device
    cudaMemcpy(d_mass, s.mass.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, s.x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, s.y.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, s.z.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, s.vx.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, s.vy.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, s.vz.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (size_t step = 0; step < nbstep; ++step) {
        compute_forces<<<gridSize, blockSize>>>(n, d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz);
        cudaDeviceSynchronize();

        update_positions<<<gridSize, blockSize>>>(n, dt, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                                  d_fx, d_fy, d_fz);
        cudaDeviceSynchronize();

        if (step % printevery == 0) {
            cudaMemcpy(s.x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.z.data(), d_z, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.vx.data(), d_vx, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.vy.data(), d_vy, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.vz.data(), d_vz, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.fx.data(), d_fx, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.fy.data(), d_fy, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(s.fz.data(), d_fz, n * sizeof(double), cudaMemcpyDeviceToHost);

            dump_state(n, s.mass.data(), s.x.data(), s.y.data(), s.z.data(),
                       s.vx.data(), s.vy.data(), s.vz.data(),
                       s.fx.data(), s.fy.data(), s.fz.data());
        }
    }

    // Free device memory
    cudaFree(d_mass); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);

    return 0;
}
