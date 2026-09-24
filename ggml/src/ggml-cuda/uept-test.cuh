#pragma once

// Explicit diagnostic entry point only; no test work is run during model inference.
// Keep the CPU oracle independent of the GPU loop: operate on a deduplicated routed union.
struct uept_test_oracle {
    std::vector<uept_entry> entries;
    std::vector<int32_t> owner;
    std::vector<uint64_t> stamp;
    std::vector<int32_t> writer;
    std::array<std::vector<const char *>, 3> ptrs;
    std::array<std::vector<char *>, 3> fill;
    uept_counters counters = {};
    uept_layer_device device;

    // Set expected host directories and empty slots for one test configuration.
    uept_test_oracle(uept_layer_device d, const std::array<std::vector<const char *>, 3> & host) :
        entries(d.n_experts, { UEPT_HOST, -1 }), owner(d.n_slots, -1), stamp(d.n_slots, 0),
        writer(d.n_experts, -1), ptrs(host), device(d) {
        for (auto & addresses : fill) {
            addresses.assign(d.n_experts, nullptr);
        }
    }

    // Pin all existing hits, then use stable LRU ordering to assign missing experts.
    void resolve(const std::vector<int32_t> & ids,
            const std::array<std::vector<const char *>, 3> & host) {
        if (counters.inflight) {
            return;
        }
        counters.inflight = 1;
        ++counters.step;
        std::fill(writer.begin(), writer.end(), -1);
        for (auto & addresses : fill) {
            std::fill(addresses.begin(), addresses.end(), nullptr);
        }
        std::vector<std::pair<int, int>> selected;
        std::vector<bool> seen(device.n_experts, false);
        for (size_t i = 0; i < ids.size(); ++i) {
            const int e = ids[i];
            counters.read_bytes += device.slot_bytes;
            if (entries[e].state == UEPT_VRAM) {
                stamp[entries[e].slot] = counters.step;
                counters.hit_bytes += device.slot_bytes;
            }
            if (!seen[e]) {
                selected.emplace_back(e, static_cast<int>(i));
                seen[e] = true;
            }
        }
        for (const auto & selection : selected) {
            const int e = selection.first;
            if (entries[e].state != UEPT_HOST) {
                continue;
            }
            std::vector<int> candidates;
            for (int s = 0; s < device.n_slots; ++s) {
                if (stamp[s] != counters.step) {
                    candidates.push_back(s);
                }
            }
            if (candidates.empty()) {
                writer[e] = -2;
                ++counters.readonly;
                continue;
            }
            const int hand = counters.hand;
            const int n_slots = device.n_slots;
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return stamp[a] != stamp[b] ? stamp[a] < stamp[b] :
                    (a - hand + n_slots) % n_slots < (b - hand + n_slots) % n_slots;
            });
            const int s = candidates.front();
            if (owner[s] >= 0) {
                const int victim = owner[s];
                entries[victim] = { UEPT_HOST, -1 };
                for (int r = 0; r < 3; ++r) {
                    ptrs[r][victim] = host[r][victim];
                }
            }
            owner[s] = e;
            stamp[s] = counters.step;
            entries[e] = { UEPT_FILLING, s };
            writer[e] = selection.second;
            for (int r = 0; r < 3; ++r) {
                fill[r][e] = device.pool + s * device.slot_bytes + device.offsets[r];
            }
            counters.hand = (s + 1) % n_slots;
            ++counters.fills;
        }
    }

    // Publishing changes addresses and state, never writer identity or weight data.
    void commit() {
        for (int e = 0; e < device.n_experts; ++e) {
            if (entries[e].state == UEPT_FILLING) {
                entries[e].state = UEPT_VRAM;
                for (int r = 0; r < 3; ++r) {
                    ptrs[r][e] = fill[r][e];
                }
            }
        }
        counters.inflight = 0;
    }

    // Compare every directory field, reverse mapping, writer and pointer against one GPU snapshot.
    bool equals(const uept_layer_device & snapshot,
            const std::array<std::vector<const char *>, 3> & host) const {
        const uept_counters & other = *snapshot.counters;
        if (other.step != counters.step || other.read_bytes != counters.read_bytes ||
                other.hit_bytes != counters.hit_bytes || other.fills != counters.fills ||
                other.readonly != counters.readonly || other.hand != counters.hand || other.inflight != counters.inflight) {
            return false;
        }
        for (int e = 0; e < device.n_experts; ++e) {
            if (snapshot.entries[e].state != entries[e].state || snapshot.entries[e].slot != entries[e].slot ||
                    snapshot.writer[e] != writer[e]) {
                return false;
            }
            for (int r = 0; r < 3; ++r) {
                if (snapshot.host[r][e] != host[r][e] || snapshot.ptrs[r][e] != ptrs[r][e] ||
                        snapshot.fill[r][e] != fill[r][e]) {
                    return false;
                }
            }
        }
        for (int s = 0; s < device.n_slots; ++s) {
            if (snapshot.owner[s] != owner[s] || snapshot.stamp[s] != stamp[s]) {
                return false;
            }
        }
        return true;
    }
};

// Test U01-U04 directly on this device for every M=1..8 and C=0..64, independently of a model.
bool ggml_backend_cuda_uept_test(ggml_backend_t backend, uint32_t seed, int repetitions) {
    GGML_ASSERT(repetitions >= 4 && repetitions <= 1000);
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    constexpr int n_experts = 96;
    constexpr int n_used = 10;
    constexpr size_t matrix_bytes = 256;
    constexpr size_t stage_stride = matrix_bytes + 416;
    uept_layer_device d = {};
    int32_t * phase_device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&phase_device), sizeof(int32_t)));
    d.phase = phase_device;
    uept_set_phase_kernel<<<1, 1, 0, ctx.stream()>>>(phase_device, 1);
    d.n_experts = n_experts;
    d.n_slots = 64;
    d.slot_bytes = 3 * matrix_bytes;
    for (int r = 0; r < 3; ++r) {
        d.offsets[r] = r * matrix_bytes;
    }
    const size_t max_bytes = uept_layer_layout(d, nullptr);
    void * allocation = nullptr;
    CUDA_CHECK(cudaMalloc(&allocation, max_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d.pool), 64 * d.slot_bytes));
    int32_t * ids_device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ids_device), 8 * n_used * sizeof(int32_t)));
    void * host_allocation = nullptr;
    void * device_alias = nullptr;
    CUDA_CHECK(hipHostMalloc(&host_allocation, 3 * n_experts * matrix_bytes, hipHostMallocMapped | hipHostMallocPortable));
    CUDA_CHECK(hipHostGetDevicePointer(&device_alias, host_allocation, 0));
    for (size_t i = 0; i < 3 * n_experts * matrix_bytes; ++i) {
        static_cast<unsigned char *>(host_allocation)[i] = static_cast<unsigned char>((i * 17 + 3) % 251);
    }
    CUDA_CHECK(cudaMemset(d.pool, 0x5a, 64 * d.slot_bytes));
    char * staging_device = nullptr;
    const char ** pointers_device = nullptr;
    int32_t * bounds_device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&staging_device), n_experts * stage_stride));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointers_device), n_experts * sizeof(const char *)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&bounds_device), (n_experts + 1) * sizeof(int32_t)));
    std::array<std::vector<const char *>, 3> host;
    for (int r = 0; r < 3; ++r) {
        for (int e = 0; e < n_experts; ++e) {
            host[r].push_back(static_cast<char *>(device_alias) + (r * n_experts + e) * matrix_bytes);
        }
    }
    std::vector<char> snapshot_bytes(max_bytes);
    uint32_t random = seed ? seed : 1;
    auto next_random = [&]() {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    uint64_t checks = 0;
    uint64_t gather_checks = 0;
    uint64_t phase_checks = 0;
    bool ok = true;
    for (int slots = 0; slots <= 64 && ok; ++slots) {
        d.n_slots = slots;
        const size_t bytes = uept_layer_layout(d, static_cast<char *>(allocation));
        for (int r = 0; r < 3; ++r) {
            CUDA_CHECK(cudaMemcpy(d.host[r], host[r].data(), n_experts * sizeof(const char *), cudaMemcpyHostToDevice));
        }
        for (int tokens = 1; tokens <= 8 && ok; ++tokens) {
            uept_reset_kernel<<<1, 256, 0, ctx.stream()>>>(d);
            uept_test_oracle oracle(d, host);
            auto compare = [&](const char * phase, int iteration) {
                CUDA_CHECK(cudaMemcpyAsync(snapshot_bytes.data(), allocation, bytes, cudaMemcpyDeviceToHost, ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                uept_layer_device snapshot = d;
                uept_layer_layout(snapshot, snapshot_bytes.data());
                ++checks;
                if (!oracle.equals(snapshot, host)) {
                    GGML_LOG_ERROR("UEPT U01-U04 mismatch: seed=%u C=%d M=%d iteration=%d phase=%s\n",
                                   seed, slots, tokens, iteration, phase);
                    return false;
                }
                return true;
            };
            ok = compare("reset", -1);
            for (int iteration = 0; iteration < repetitions && ok; ++iteration) {
                std::vector<int32_t> ids(tokens * n_used);
                for (size_t i = 0; i < ids.size(); ++i) {
                    // Mix repeated IDs, high-cardinality unions and random hit/miss order.
                    ids[i] = iteration % 4 == 0 ? int(i % 3) :
                             iteration % 4 == 1 ? int((i + iteration * 13) % n_experts) :
                             int(next_random() % n_experts);
                }
                if (iteration % 4 == 3 && slots > 0 && oracle.owner[0] >= 0) {
                    // A late hit must remain pinned even when earlier misses need its old LRU slot.
                    ids.back() = oracle.owner[0];
                }
                CUDA_CHECK(cudaMemcpyAsync(ids_device, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
                uept_resolve_kernel<<<1, 32, 0, ctx.stream()>>>(d, ids_device, tokens, n_used, n_used);
                oracle.resolve(ids, host);
                ok = compare("resolve", iteration);
                if (!ok) {
                    break;
                }
                // Separate up/gate launches must observe exactly the same resolve result.
                uept_resolve_kernel<<<1, 32, 0, ctx.stream()>>>(d, ids_device, tokens, n_used, n_used);
                ok = compare("repeated-resolve", iteration);
                if (!ok) {
                    break;
                }
                uept_commit_kernel<<<1, 256, 0, ctx.stream()>>>(d);
                oracle.commit();
                ok = compare("commit", iteration);
                if (ok && iteration + 1 == repetitions) {
                    // A prefill gather must not update even one byte of the persistent directory.
                    const std::vector<char> before_gather = snapshot_bytes;
                    const int role = (tokens + slots) % 3;
                    std::vector<int32_t> bounds(n_experts + 1, 0);
                    for (int expert : ids) {
                        ++bounds[expert + 1];
                    }
                    for (int e = 0; e < n_experts; ++e) {
                        bounds[e + 1] += bounds[e];
                    }
                    CUDA_CHECK(cudaMemcpyAsync(bounds_device, bounds.data(), bounds.size() * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, ctx.stream()));
                    // Poison every tail on every call, including reuse with a different role.
                    CUDA_CHECK(cudaMemsetAsync(staging_device, 0xff, n_experts * stage_stride, ctx.stream()));
                    const dim3 blocks(1, n_experts, 1);
                    uept_gather_kernel<<<blocks, 256, 0, ctx.stream()>>>(d, role, bounds_device,
                        staging_device, matrix_bytes, stage_stride, pointers_device);
                    ++gather_checks;
                    std::vector<const char *> pointers(n_experts);
                    std::vector<unsigned char> staging(n_experts * stage_stride);
                    CUDA_CHECK(cudaMemcpyAsync(pointers.data(), pointers_device, n_experts * sizeof(const char *),
                                               cudaMemcpyDeviceToHost, ctx.stream()));
                    CUDA_CHECK(cudaMemcpyAsync(staging.data(), staging_device, staging.size(),
                                               cudaMemcpyDeviceToHost, ctx.stream()));
                    ok = compare("gather-directory", iteration);
                    ok = ok && memcmp(before_gather.data(), snapshot_bytes.data(), bytes) == 0;
                    for (int e = 0; e < n_experts && ok; ++e) {
                        const bool copied = bounds[e + 1] > bounds[e] && oracle.entries[e].state != UEPT_VRAM;
                        const char * expected_pointer = copied ? staging_device + e * stage_stride : oracle.ptrs[role][e];
                        ok = pointers[e] == expected_pointer;
                        for (size_t i = 0; i < stage_stride && ok; ++i) {
                            const unsigned char expected = !copied ? 0xff : i >= matrix_bytes ? 0 :
                                static_cast<unsigned char *>(host_allocation)[(role * n_experts + e) * matrix_bytes + i];
                            ok = staging[e * stage_stride + i] == expected;
                        }
                    }
                    if (!ok) {
                        GGML_LOG_ERROR("UEPT gather invariants failed: seed=%u C=%d M=%d role=%d\n", seed, slots, tokens, role);
                    }
                    if (ok) {
                        // Even a one-token prompt must preserve warm directory bytes, writers and counters.
                        const std::vector<char> before_prefill = snapshot_bytes;
                        uept_set_phase_kernel<<<1, 1, 0, ctx.stream()>>>(phase_device, 0);
                        uept_resolve_kernel<<<1, 32, 0, ctx.stream()>>>(d, ids_device, tokens, n_used, n_used);
                        uept_commit_kernel<<<1, 256, 0, ctx.stream()>>>(d);
                        ok = compare("prefill-readonly", iteration);
                        ok = ok && memcmp(before_prefill.data(), snapshot_bytes.data(), bytes) == 0;
                        ++phase_checks;
                        uept_set_phase_kernel<<<1, 1, 0, ctx.stream()>>>(phase_device, 1);
                        if (!ok) {
                            GGML_LOG_ERROR("UEPT prefill phase mutated directory: C=%d M=%d\n", slots, tokens);
                        }
                    }
                }
            }
        }
    }
    if (ok) {
        // Confirm that Release validation detects corruption and request reset cannot hide its sticky error.
        uept_layer_device * check_device = nullptr;
        uept_check_failure * error_host = nullptr;
        uept_check_failure * error_device = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&check_device), sizeof(d)));
        CUDA_CHECK(cudaMemcpy(check_device, &d, sizeof(d), cudaMemcpyHostToDevice));
        CUDA_CHECK(hipHostMalloc(reinterpret_cast<void **>(&error_host), sizeof(uept_check_failure),
                                hipHostMallocMapped | hipHostMallocPortable));
        CUDA_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&error_device), error_host, 0));
        memset(error_host, 0, sizeof(uept_check_failure));
        uept_validate_kernel<<<1, 256, 0, ctx.stream()>>>(check_device, error_device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        ok = error_host->code == 0;
        const uept_entry invalid = { UEPT_FILLING, -1 };
        CUDA_CHECK(cudaMemcpyAsync(d.entries, &invalid, sizeof(invalid), cudaMemcpyHostToDevice, ctx.stream()));
        uept_validate_kernel<<<1, 256, 0, ctx.stream()>>>(check_device, error_device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        ok = ok && error_host->code == 2 && error_host->index == 0;
        uept_reset_kernel<<<1, 256, 0, ctx.stream()>>>(d);
        uept_validate_kernel<<<1, 256, 0, ctx.stream()>>>(check_device, error_device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        ok = ok && error_host->code == 2 && error_host->index == 0;
        // This synchronized reinitialization emulates a fresh diagnostic allocation.
        memset(error_host, 0, sizeof(uept_check_failure));
        uept_validate_kernel<<<1, 256, 0, ctx.stream()>>>(check_device, error_device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        ok = ok && error_host->code == 0;
        if (!ok) {
            GGML_LOG_ERROR("UEPT Release validation fault/reset diagnostic failed\n");
        }
        CUDA_CHECK(cudaFree(check_device));
        CUDA_CHECK(hipHostFree(error_host));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaFree(ids_device));
    CUDA_CHECK(cudaFree(phase_device));
    CUDA_CHECK(cudaFree(staging_device));
    CUDA_CHECK(cudaFree(pointers_device));
    CUDA_CHECK(cudaFree(bounds_device));
    CUDA_CHECK(cudaFree(d.pool));
    CUDA_CHECK(cudaFree(allocation));
    CUDA_CHECK(hipHostFree(host_allocation));
    GGML_LOG_INFO("UEPT U01-U04: result=%s seed=%u configurations=%d checks=%llu gather_checks=%llu phase_checks=%llu validation_fault_checks=4 repetitions=%d\n",
                  ok ? "PASS" : "FAIL", seed, 65 * 8, (unsigned long long) checks,
                  (unsigned long long) gather_checks, (unsigned long long) phase_checks, repetitions);
    return ok;
}
