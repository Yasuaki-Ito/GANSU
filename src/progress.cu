/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 */

#include "progress.hpp"

namespace gansu {

static thread_local void (*g_progress_fn)(const char*, int, int, const double*, void*) = nullptr;
static thread_local void* g_progress_userdata = nullptr;

void set_progress_callback(void (*fn)(const char*, int, int, const double*, void*), void* user_data) {
    g_progress_fn = fn;
    g_progress_userdata = user_data;
}

void clear_progress_callback() {
    g_progress_fn = nullptr;
    g_progress_userdata = nullptr;
}

void report_progress(const char* stage, int iter, int n_values, const double* values) {
    if (g_progress_fn) {
        g_progress_fn(stage, iter, n_values, values, g_progress_userdata);
    }
}

} // namespace gansu
