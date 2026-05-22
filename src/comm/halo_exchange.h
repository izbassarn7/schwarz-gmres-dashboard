#pragma once

#include <mpi.h>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

namespace schwarz {

// MPI halo exchange for distributed domain decomposition.
// Each rank owns a contiguous range of global DOFs and needs
// ghost values from neighboring ranks.
struct HaloExchange {
    int rank = 0;
    int nranks = 0;
    MPI_Comm comm = MPI_COMM_NULL;

    // send_map[neighbor_rank] = list of LOCAL indices this rank sends
    std::map<int, std::vector<int>> send_map;
    // recv_map[neighbor_rank] = list of LOCAL indices to write received data into
    std::map<int, std::vector<int>> recv_map;

    // Persistent buffers to avoid allocation per exchange
    mutable std::map<int, std::vector<double>> send_bufs;
    mutable std::map<int, std::vector<double>> recv_bufs;

    // Setup the halo exchange pattern.
    //   global_to_local: maps global DOF -> local index on THIS rank
    //   ghost_global_ids: global DOF IDs that this rank needs but does not own
    //   owner_of_global: function/map giving owning rank for any global DOF
    void setup(MPI_Comm c,
               const std::vector<int>& owned_global_ids,
               const std::vector<int>& ghost_global_ids,
               const std::vector<int>& ghost_owners,
               const std::map<int, int>& global_to_local)
    {
        comm = c;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nranks);
        send_map.clear();
        recv_map.clear();

        // Build recv_map: for each ghost DOF, record which rank owns it
        // and the local index where we'll store the received value
        for (size_t i = 0; i < ghost_global_ids.size(); ++i) {
            int owner = ghost_owners[i];
            int gid = ghost_global_ids[i];
            auto it = global_to_local.find(gid);
            if (it != global_to_local.end() && owner != rank)
                recv_map[owner].push_back(it->second);
        }

        // Exchange: tell each neighbor which global DOFs we need from them.
        // Then they respond with the local indices they'll send.

        // Step 1: send the count of DOFs we need from each rank
        std::vector<int> recv_counts(nranks, 0);
        for (auto& [r, indices] : recv_map)
            recv_counts[r] = static_cast<int>(indices.size());

        std::vector<int> send_counts(nranks, 0);
        MPI_Alltoall(recv_counts.data(), 1, MPI_INT,
                     send_counts.data(), 1, MPI_INT, comm);

        // Step 2: send the actual global DOF IDs we need
        // Build send/recv displacements
        std::vector<int> sdispls(nranks, 0), rdispls(nranks, 0);
        for (int i = 1; i < nranks; ++i) {
            sdispls[i] = sdispls[i - 1] + recv_counts[i - 1];
            rdispls[i] = rdispls[i - 1] + send_counts[i - 1];
        }
        int total_send_ids = sdispls[nranks - 1] + recv_counts[nranks - 1];
        int total_recv_ids = rdispls[nranks - 1] + send_counts[nranks - 1];

        // Pack the global IDs we need into a flat send buffer
        std::vector<int> send_gids(total_send_ids);
        for (auto& [r, indices] : recv_map) {
            int off = sdispls[r];
            for (size_t i = 0; i < indices.size(); ++i) {
                // Reverse-lookup: local_index -> global_id
                for (auto& [gid, lid] : global_to_local) {
                    if (lid == indices[i]) {
                        send_gids[off + static_cast<int>(i)] = gid;
                        break;
                    }
                }
            }
        }

        std::vector<int> recv_gids(total_recv_ids);
        MPI_Alltoallv(send_gids.data(), recv_counts.data(), sdispls.data(), MPI_INT,
                      recv_gids.data(), send_counts.data(), rdispls.data(), MPI_INT,
                      comm);

        // Step 3: build send_map from the received global IDs
        // These are the DOFs that neighbors need from us
        for (int r = 0; r < nranks; ++r) {
            if (send_counts[r] == 0) continue;
            auto& slist = send_map[r];
            slist.reserve(send_counts[r]);
            for (int i = 0; i < send_counts[r]; ++i) {
                int gid = recv_gids[rdispls[r] + i];
                auto it = global_to_local.find(gid);
                if (it != global_to_local.end())
                    slist.push_back(it->second);
            }
        }

        // Allocate persistent buffers
        for (auto& [r, indices] : send_map)
            send_bufs[r].resize(indices.size());
        for (auto& [r, indices] : recv_map)
            recv_bufs[r].resize(indices.size());
    }

    // Non-blocking halo exchange on host data.
    // local_data must have space for owned + ghost DOFs.
    void exchange(double* local_data) const {
        std::vector<MPI_Request> requests;
        requests.reserve(recv_map.size() + send_map.size());

        // Post receives
        for (auto& [src, indices] : recv_map) {
            auto& buf = recv_bufs[src];
            MPI_Request req;
            MPI_Irecv(buf.data(), static_cast<int>(buf.size()),
                      MPI_DOUBLE, src, 0, comm, &req);
            requests.push_back(req);
        }

        // Pack and send
        for (auto& [dst, indices] : send_map) {
            auto& buf = send_bufs[dst];
            for (size_t i = 0; i < indices.size(); ++i)
                buf[i] = local_data[indices[i]];
            MPI_Request req;
            MPI_Isend(buf.data(), static_cast<int>(buf.size()),
                      MPI_DOUBLE, dst, 0, comm, &req);
            requests.push_back(req);
        }

        MPI_Waitall(static_cast<int>(requests.size()),
                     requests.data(), MPI_STATUSES_IGNORE);

        // Unpack received data into ghost positions
        for (auto& [src, indices] : recv_map) {
            auto& buf = recv_bufs[src];
            for (size_t i = 0; i < indices.size(); ++i)
                local_data[indices[i]] = buf[i];
        }
    }
};

}  // namespace schwarz
