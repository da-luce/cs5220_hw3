#include "common.h"
#include <mpi.h>

// Put any static global variables here that you will use throughout the simulation.

#include "common.h"
#include <cmath>
#include <vector>
#include <algorithm>

// I use linked-lists instead of vectors for bins because it's much faster to
// prepend to a linked list than to append to a vector. This is a common technique
// and based by implementation off this writeup https://aiichironakano.github.io/cs596/01-1LinkedListCell.pdf
// Google Gemini was used to help discover this approach and also conceputalize why it is better

static int num_bins_per_dim; // Number of bins along one dimension
static int total_bins; // Total number of bins in the grid
static int* bin_heads = nullptr; // Array of bin heads for linked list
static int* next_particle = nullptr; // Array of next particle indices for linked list
#define LIST_END -1

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    // Apply an equal and opposite force to neighbor
    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// I adopted this approach as it is a familier "LeetCode" algorithm style on grid problems (like connect4)
static std::pair<int, int> directions[] = {
    // {Drow, Dcol}
    {0, 1},  // E
    {1, -1}, // SW
    {1, 0},  // S
    {1, 1},   // SE

    // We don't need to double count, so we only expand to the bottom-right direction
};

void init_simulation_serial(particle_t* parts, int num_parts, double size) {
    // Resize a grid bin to be the size of the cutoff
    num_bins_per_dim = ceil(size / cutoff);
    total_bins = num_bins_per_dim * num_bins_per_dim;

    // Allocate memory for bin heads and linked list of particles if not already allocated
    if (bin_heads == nullptr) {
        bin_heads = new int[total_bins];
        next_particle = new int[num_parts];
    }
}

void set_bin_coordinates(double x, double y, double size, int& bx, int& by) {
    bx = int(x / cutoff);
    by = int(y / cutoff);

    // Safety clamp in case a particle is exactly on the upper boundary
    if (bx < 0) {
        bx = 0;
    } else if (bx >= num_bins_per_dim) {
        bx = num_bins_per_dim - 1;
    }

    if (by < 0) {
        by = 0;
    } else if (by >= num_bins_per_dim) {
        by = num_bins_per_dim - 1;
    }
}

void simulate_one_step_serial(particle_t* parts, int num_parts, double size) {

    // Clear grid: Initialize all heads to empty
    for (int i = 0; i < total_bins; ++i) {
        bin_heads[i] = LIST_END;
    }

    // Reset accelerations
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = 0;
        parts[i].ay = 0;
    }

    // Compute which particles are in which bins
    for (int i = 0; i < num_parts; ++i) {

        int bx, by;
        set_bin_coordinates(parts[i].x, parts[i].y, size, bx, by);
        int bin_index = by * num_bins_per_dim + bx;

        // Add particle (index) to the head of bin linked list
        next_particle[i] = bin_heads[bin_index];
        bin_heads[bin_index] = i;
    }

    // Compute forces
    for (int by = 0; by < num_bins_per_dim; ++by) {
        for (int bx = 0; bx < num_bins_per_dim; ++bx) {
            int bin_index = by * num_bins_per_dim + bx;

            // Check inside same bin
            // Notice the tail is always LIST_END
            for (int i = bin_heads[bin_index]; i != LIST_END; i = next_particle[i]) {
                for (int j = next_particle[i]; j != LIST_END; j = next_particle[j]) {
                    apply_force(parts[i], parts[j]);
                }
            }

            for (auto dir : directions) {
                int dy = dir.first;
                int dx = dir.second;

                int ny = by + dy;
                int nx = bx + dx;

                if (ny < 0 || ny >= num_bins_per_dim || nx < 0 || nx >= num_bins_per_dim) {
                    continue;
                }

                int nbin_index = ny * num_bins_per_dim + nx;

                // Compute forces between current bin and neighbor bin
                for (int i = bin_heads[bin_index]; i != LIST_END; i = next_particle[i]) {
                    for (int j = bin_heads[nbin_index]; j != LIST_END; j = next_particle[j]) {
                        apply_force(parts[i], parts[j]);
                    }
                }
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    init_simulation_serial(parts, num_parts, size);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    simulate_one_step_serial(parts, num_parts, size);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    if (num_procs == 1) {
        // Rank 0 already has all particles. Just sort them by ID.
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id; // Assumes your particle_t struct has an 'id' field
        });
        return;
    }
}
