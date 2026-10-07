#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <map>
#include <memory>

struct llama_model;

// cumulative MoE cache counters
struct llama_moe_cache_stats {
    size_t hits   = 0;
    size_t misses = 0;
    size_t masked = 0; // experts dropped instead of uploaded
    size_t bytes  = 0;
};

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// MUL_MAT_ID ops on these experts run on the device and only the cache misses are uploaded
class llama_moe_cache {
public:
    // protect_top experts from the top of the gate order are never dropped
    // min_experts is the number of experts a token computes at least, it caps how many experts can be dropped
    llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size,
        int32_t protect_top, int32_t min_experts);
    ~llama_moe_cache();

    ggml_backend_t backend() const;

    // the resolve() calls that follow belong to a batch that checks the tokens proposed by a draft
    // such a batch does not use the cache and computes its experts on the host
    // the flag stays set until the next call, so reset it with verify_guard
    void set_verify(bool verify);

    // keeps the flag set for the lifetime of the guard, then clears it
    // the cache is disabled for all batches if a set_verify(true) is not matched by a reset
    class verify_guard {
    public:
        verify_guard(llama_moe_cache & cache, bool verify) : cache(cache) {
            cache.set_verify(verify);
        }

        ~verify_guard() {
            cache.set_verify(false);
        }

        verify_guard(const verify_guard &) = delete;
        verify_guard & operator=(const verify_guard &) = delete;

    private:
        llama_moe_cache & cache;
    };

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // log the counters accumulated since the previous call
    // bucket 0 is ubatch up to 8 tokens, bucket 1 is the rest
    void log_turn_stats() const;

    // ggml_backend_sched callbacks, user_data is the llama_moe_cache
    static bool sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry);
    static void sched_begin(void * user_data);
    static bool sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
