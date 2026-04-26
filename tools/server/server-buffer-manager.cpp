#include "server-buffer-manager.h"

#include "ggml.h"

SlotBufferState& ResponseBufferManager::get_or_create_buffer(int slot_id) {
    auto it = slots_buffers.find(slot_id);
    if (it == slots_buffers.end()) {
        slots_buffers[slot_id] = std::make_unique<SlotBufferState>();
    }
    return *slots_buffers[slot_id];
}

completion_token_output ResponseBufferManager::merge_token_outputs(
    std::vector<completion_token_output>& buffer,
    const completion_token_output& current) {
    
    completion_token_output merged;
    merged.tok = current.tok;
    merged.text_to_send = "";
    merged.prob = current.prob;
    
    for (const auto& buf : buffer) {
        merged.text_to_send += buf.text_to_send;
    }
    if (!current.text_to_send.empty()) {
        merged.text_to_send += current.text_to_send;
    }
    
    for (const auto& buf : buffer) {
        merged.probs.insert(merged.probs.end(), buf.probs.begin(), buf.probs.end());
    }
    merged.probs.insert(merged.probs.end(), current.probs.begin(), current.probs.end());
    
    return merged;
}

void ResponseBufferManager::submit_result(
    int slot_id,
    const completion_token_output& result,
    std::function<void(const completion_token_output&)> send_callback) {
    
    if (!is_enabled(slot_id)) {
        send_callback(result);
        return;
    }
    
    SlotBufferState& state = get_or_create_buffer(slot_id);
    const int64_t current_time = ggml_time_us();
    
    if (state.is_first_token) {
        send_callback(result);
        state.t_last_send = current_time;
        state.is_first_token = false;
        return;
    }
    
    const int64_t elapsed_ms = (current_time - state.t_last_send) / 1000;
    
    if (elapsed_ms >= buffer_interval_ms) {
        if (!state.buffer.empty()) {
            completion_token_output merged = merge_token_outputs(state.buffer, result);
            send_callback(merged);
            state.buffer.clear();
        } else {
            send_callback(result);
        }
        state.t_last_send = current_time;
    } else {
        state.buffer.push_back(result);
    }
}

void ResponseBufferManager::flush_slot(
    int slot_id,
    std::function<void(const completion_token_output&)> send_callback) {
    
    if (!is_enabled(slot_id)) {
        return;
    }
    
    SlotBufferState& state = get_or_create_buffer(slot_id);
    
    if (!state.buffer.empty()) {
        completion_token_output empty;
        empty.tok = 0;
        empty.text_to_send = "";
        empty.prob = 0.0f;
        
        completion_token_output merged = merge_token_outputs(state.buffer, empty);
        send_callback(merged);
        state.buffer.clear();
    }
    
    state.t_last_send = ggml_time_us();
}

void ResponseBufferManager::reset_slot(int slot_id) {
    auto it = slots_buffers.find(slot_id);
    if (it != slots_buffers.end()) {
        it->second->reset();
    }
}

void ResponseBufferManager::set_enabled(int slot_id, bool enabled) {
    SlotBufferState& state = get_or_create_buffer(slot_id);
    state.enabled = enabled;
}

bool ResponseBufferManager::is_enabled(int slot_id) const {
    auto it = slots_buffers.find(slot_id);
    if (it == slots_buffers.end()) {
        return false;
    }
    return it->second->enabled;
}
