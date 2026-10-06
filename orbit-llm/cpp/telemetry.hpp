// Process and machine memory telemetry (Windows), shared by orbit_run and
// orbit_chat.
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <cstdint>

namespace orbit {

inline double bytes_to_mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// Working set: pages of this process currently in RAM, including the
// memory-mapped model file.
inline double get_rss_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.WorkingSetSize);
}

inline double get_lifetime_peak_rss_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.PeakWorkingSetSize);
}

// Committed private memory. Unlike the working set, this includes memory that
// has been allocated but not yet touched (e.g. a pre-allocated KV cache), and it
// is what counts against the system commit limit when predicting OOM.
inline double get_commit_mib() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    GetProcessMemoryInfo(GetCurrentProcess(),
                         reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                         sizeof(counters));
    return bytes_to_mib(counters.PrivateUsage);
}

inline double get_lifetime_peak_commit_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.PeakPagefileUsage);
}

inline double get_available_ram_mib() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return bytes_to_mib(status.ullAvailPhys);
}

// Power source. On battery this laptop runs inference roughly 10x slower
// (FINDINGS Finding 20), so every measurement records it.
struct PowerState {
    bool known = false;
    bool on_ac = false;
    int battery_percent = -1;  // -1 = unknown or no battery
};

inline PowerState get_power_state() {
    SYSTEM_POWER_STATUS status{};
    PowerState state;
    if (GetSystemPowerStatus(&status)) {
        state.known = status.ACLineStatus != 255;
        state.on_ac = status.ACLineStatus == 1;
        state.battery_percent = status.BatteryLifePercent == 255 ? -1 : status.BatteryLifePercent;
    }
    return state;
}

inline double get_total_ram_mib() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return bytes_to_mib(status.ullTotalPhys);
}

}  // namespace orbit
