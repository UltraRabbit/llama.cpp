#pragma once

#include "server-task.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>

struct SlotBufferState {
    std::vector<completion_token_output> buffer;
    int64_t t_last_send = 0;
    bool is_first_token = true;
    bool enabled = false;
    
    void reset() {
        buffer.clear();
        t_last_send = 0;
        is_first_token = true;
        enabled = false;
    }
};

class ResponseBufferManager {
private:
    std::unordered_map<int, std::unique_ptr<SlotBufferState>> slots_buffers;
    
    const int64_t buffer_interval_ms = 300;
    
    SlotBufferState& get_or_create_buffer(int slot_id);
    
    completion_token_output merge_token_outputs(
        std::vector<completion_token_output>& buffer,
        const completion_token_output& current);
    
public:
    void submit_result(
        int slot_id,
        const completion_token_output& result,
        std::function<void(const completion_token_output&)> send_callback);
    
    void flush_slot(
        int slot_id,
        std::function<void(const completion_token_output&)> send_callback);
    
    void reset_slot(int slot_id);
    
    void set_enabled(int slot_id, bool enabled);
    
    bool is_enabled(int slot_id) const;
};
