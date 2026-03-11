#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>


static double bin_size = cutoff;

static int num_bins_per_dim; // Calculated in init_simulation based on size and cutoff

static double slice_height; // Calculated values for dimensions of this slice
static double local_y_min;
static double local_y_max;

static int local_start_by; // The range of bins we are responsible for (inclusive)
static int local_end_by;
static int local_num_rows;

static MPI_Datatype MPI_PARTICLE;           // MPI datatype for particle_t
static std::vector<particle_t> local_parts; // All local particles (including ghosts) for this process

// Persistent buffers to prevent memory reallocation every frame
static std::vector<std::vector<int>> bins; 
static std::vector<particle_t> tx_up_buf, tx_down_buf;
static std::vector<particle_t> rx_up_buf, rx_down_buf;
static std::vector<particle_t> move_up_buf, move_down_buf;
static std::vector<particle_t> incoming_parts; // Used for both ghost AND migrant exchanges;

// Use same direction approach as in hw2
static std::pair<int, int> directions[] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 1},  
    {1, -1},
    {1, 0},
    {1, 1} 
};

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
}

// Apply symmetric force for particles in the same bin
void apply_symmetric_force(particle_t& particle, particle_t& neighbor) {
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

    // Apply equal and opposite force to the neighbor
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

// Helper to exchange variable-length vectors using the static buffers
void exchange_particles(std::vector<particle_t>& send_up, std::vector<particle_t>& send_down, 
                        std::vector<particle_t>& received, int rank, int num_procs) {

    int count_send_up = send_up.size();
    int count_send_down = send_down.size();
    int count_recv_up = 0, count_recv_down = 0;

    // Used Google Gemini to help understand the use of tags in order to avoid deadlock, since we
    // didn't really discuss tags in lecture. Here, tags are used to differentiate
    // between the two directions of communication.

    MPI_Request reqs[4]; // (Up + Down) x (Send + Recv) = 4 possible operations
    int req_cnt = 0;

    // 1. Exchange counts of particles to send
    if (rank > 0) {
        MPI_Isend(&count_send_up, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]); // Number to send up
        MPI_Irecv(&count_recv_up, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]); // Number to receive from up
    }
    if (rank < num_procs - 1) {
        MPI_Isend(&count_send_down, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_cnt++]); // Number to send down
        MPI_Irecv(&count_recv_down, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_cnt++]); // Number to receive from down
    }
    MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE); // Wait for counts to be exchanged before resizing buffers

    // 2. Resize receive buffers
    rx_up_buf.resize(count_recv_up);
    rx_down_buf.resize(count_recv_down);
    req_cnt = 0;

    // 3. Exchange particle data
    if (rank > 0) {
        if (count_send_up > 0) {
            // Send to rank - 1
            MPI_Isend(send_up.data(), count_send_up, MPI_PARTICLE, rank - 1, 2, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        if (count_recv_up > 0) {
            // Receive from rank - 1
            MPI_Irecv(rx_up_buf.data(), count_recv_up, MPI_PARTICLE, rank - 1, 3, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
    }

    if (rank < num_procs - 1) {
        if (count_send_down > 0) {
            // Send to rank + 1
            MPI_Isend(send_down.data(), count_send_down, MPI_PARTICLE, rank + 1, 3, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        if (count_recv_down > 0) {
            // Receive from rank + 1
            MPI_Irecv(rx_down_buf.data(), count_recv_down, MPI_PARTICLE, rank + 1, 2, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
    }
    MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);

    // 4. Append to output buffer
    received.clear();
    received.insert(received.end(), rx_up_buf.begin(), rx_up_buf.end());
    received.insert(received.end(), rx_down_buf.begin(), rx_down_buf.end());
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // 1. Exchange Ghosts
    tx_up_buf.clear();
    tx_down_buf.clear();

    for (const auto& p : local_parts) {
        // If a particle is within `cutoff` from our border, it is a ghost to the 
        // neighboring process!

        if (rank > 0 && p.y < local_y_min + cutoff) {
            tx_up_buf.push_back(p);
        }
        if (rank < num_procs - 1 && p.y > local_y_max - cutoff) {
            tx_down_buf.push_back(p);
        }
    }

    // Append ghosts directly for easy binning
    // If the index of a particle is >= num_local, it is a ghost!
    int num_local = local_parts.size();
    exchange_particles(tx_up_buf, tx_down_buf, incoming_parts, rank, num_procs);
    local_parts.insert(local_parts.end(), incoming_parts.begin(), incoming_parts.end());
    int total_parts = local_parts.size();

    // 2. Clear bins
    for (auto& bin : bins){
        bin.clear();
    }

    // 3. Clear accelerations
    for (int i = 0; i < num_local; ++i) {
        local_parts[i].ax = 0;
        local_parts[i].ay = 0;
    }

    // 4. Bin particles
    int y_offset = std::max(0, local_start_by - 1); // Account for the top ghost row

    for (int i = 0; i < total_parts; ++i) {
        int bx = std::max(0, std::min((int)(local_parts[i].x / cutoff), num_bins_per_dim - 1));
        int by = std::max(0, std::min((int)(local_parts[i].y / cutoff), num_bins_per_dim - 1));

        int local_by = by - y_offset;
        if (local_by >= 0 && local_by < local_num_rows) {
            bins[local_by * num_bins_per_dim + bx].push_back(i);
        }
    }

    // 5. Compute forces
    for (int by = local_start_by; by <= local_end_by; ++by) {
        int local_by = by - y_offset;
        for (int bx = 0; bx < num_bins_per_dim; ++bx) {

            // We have selected a bin globally at (bx, by), relatively at (bx, local_by)

            int bin_index = local_by * num_bins_per_dim + bx;
            auto& current_bin = bins[bin_index];

            // Same bin interactions
            for (size_t i = 0; i < current_bin.size(); ++i) {
                int p1_idx = current_bin[i];

                // Do not apply forces from ghosts to other particles
                if (p1_idx >= num_local) {
                    continue;
                }

                for (size_t j = i + 1; j < current_bin.size(); ++j) {
                    int p2_idx = current_bin[j];
                    if (p2_idx < num_local) {
                        apply_symmetric_force(local_parts[p1_idx], local_parts[p2_idx]);
                    } else {

                        // We apply one direction of force on ghosts since they won't be applying forces back on us
                        apply_force(local_parts[p1_idx], local_parts[p2_idx]);
                    }
                }
            }

            // Neighbor bin interactions
            for (auto dir : directions) {
                int nx = bx + dir.second;
                int ny = by + dir.first;
                int n_local_by = ny - y_offset;

                // If the neighbor bin is out of bounds, skip it
                if (nx < 0 || nx >= num_bins_per_dim || n_local_by < 0 || n_local_by >= local_num_rows) {
                    continue;
                }

                int nbin_index = n_local_by * num_bins_per_dim + nx;
                for (int p1_idx : current_bin) {

                    // Again do not apply forces from ghosts to other particles
                    if (p1_idx >= num_local) {
                        continue;
                    }

                    for (int p2_idx : bins[nbin_index]) {
                        apply_force(local_parts[p1_idx], local_parts[p2_idx]);
                    }
                }
            }
        }
    }

    // 6. Strip ghosts 
    local_parts.resize(num_local); // Drops all ghosts off back of vector

    // 7. Move locally owned particles
    for (int i = 0; i < num_local; ++i) {
        move(local_parts[i], size);
    }

    // 8. Migrate particles that have moved
    move_up_buf.clear();
    move_down_buf.clear();

    // The order of the vector doesn't matter so it is more efficient to drop the end and swap
    for (int i = local_parts.size() - 1; i >= 0; --i) {
        int owner_rank = std::min((int)(local_parts[i].y / slice_height), num_procs - 1);
        if (owner_rank < rank) {
            move_up_buf.push_back(local_parts[i]);
            local_parts[i] = local_parts.back();
            local_parts.pop_back();
        } else if (owner_rank > rank) {
            move_down_buf.push_back(local_parts[i]);
            local_parts[i] = local_parts.back();
            local_parts.pop_back();
        }
    }

    exchange_particles(move_up_buf, move_down_buf, incoming_parts, rank, num_procs);
    local_parts.insert(local_parts.end(), incoming_parts.begin(), incoming_parts.end()); // Add incoming particles
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // Define the particle_t type for MPI
    MPI_Type_contiguous(sizeof(particle_t), MPI_BYTE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    num_bins_per_dim = ceil(size / cutoff);
    slice_height = size / num_procs;
    local_y_min = rank * slice_height;
    local_y_max = (rank + 1) * slice_height;
    if (rank == num_procs - 1) {
        local_y_max = size;
    }

    local_start_by = std::max(0, (int) (local_y_min / cutoff));
    local_end_by = std::min(num_bins_per_dim - 1, (int) (local_y_max / cutoff));

    // +2 for ghost rows above and below our slice
    local_num_rows = (local_end_by - local_start_by + 1) + 2;
    bins.resize(local_num_rows * num_bins_per_dim);

    local_parts.clear();

    // I have found in hw3 that using vectors over linked lists incurs a big overhead.
    // But, we need vectors for memory contiguousness for messaging with MPI. To avoid the big overhead
    // of reallocation, we can just reserve a good amount.
    int estimated_load = (num_parts / num_procs) * 1.5; // 50% buffer
    local_parts.reserve(estimated_load);
    tx_up_buf.reserve(estimated_load / 10); // Assume fewer particles are near border
    tx_down_buf.reserve(estimated_load / 10);
    rx_up_buf.reserve(estimated_load / 10);
    rx_down_buf.reserve(estimated_load / 10);
    move_up_buf.reserve(estimated_load / 10);
    move_down_buf.reserve(estimated_load / 10);
    incoming_parts.reserve(estimated_load / 5); // Assume some more migrants than ghosts

    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].y >= local_y_min && parts[i].y < local_y_max) {
            local_parts.push_back(parts[i]);
        } else if (rank == num_procs - 1 && parts[i].y == size) { 
            local_parts.push_back(parts[i]);
        }
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // Gather the number of particles from each process
    int local_count = local_parts.size();
    std::vector<int> recv_counts(num_procs);
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Track where each segment of particles starts
    std::vector<int> displacements(num_procs, 0);
    if (rank == 0) {
        for (int i = 1; i < num_procs; ++i) {
            displacements[i] = displacements[i-1] + recv_counts[i-1];
        }
    }

    // Do the actual gather of particles
    MPI_Gatherv(local_parts.data(), local_count, MPI_PARTICLE,
                parts, recv_counts.data(), displacements.data(), MPI_PARTICLE,
                0, MPI_COMM_WORLD);

    // Sort for rank = 0
    // I tried a distributed sort then merge on rank 0, but ended up having poorer PE
    // (though sometimes better absolute runtime)
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });
    }
}
