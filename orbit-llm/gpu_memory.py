"""GPU memory of one process, from the Windows "GPU Process Memory" counters.

iGPU buffers (weights, KV cache) live in shared system memory and are only
partly visible in the process working set. These counters report what the
graphics driver has allocated for the process:
    \\GPU Process Memory(pid_<PID>_luid_*_phys_*)\\Shared Usage
summed over all instances of the PID.

    sampler = GpuMemorySampler(pid, interval_s=1.0); sampler.start(); ...; peak = sampler.stop()
"""

import ctypes
import threading
from ctypes import wintypes

pdh = ctypes.WinDLL("pdh")
# PDH status codes are unsigned (PDH_MORE_DATA = 0x800007D2); the ctypes default is signed int.
for _name in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
              "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
    getattr(pdh, _name).restype = wintypes.DWORD
PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2
ERROR_SUCCESS = 0


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", PDH_FMT_COUNTERVALUE)]


class GpuMemoryReader:
    """Reads one process's shared GPU memory in MiB."""

    def __init__(self, pid, counter="Shared Usage"):
        self.query = wintypes.HANDLE()
        self.counter = wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != ERROR_SUCCESS:
            raise OSError("PdhOpenQueryW failed")
        path = f"\\GPU Process Memory(pid_{pid}_*)\\{counter}"
        status = pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(self.counter))
        if status != ERROR_SUCCESS:
            raise OSError(f"PdhAddEnglishCounterW failed: {status:#x}")

    def read_mib(self):
        # A wildcard instance list is refreshed on each collection.
        if pdh.PdhCollectQueryData(self.query) != ERROR_SUCCESS:
            return None
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, ctypes.byref(size),
                                                  ctypes.byref(count), None)
        if status != PDH_MORE_DATA or size.value == 0:
            return None
        buffer = (ctypes.c_byte * size.value)()
        status = pdh.PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, ctypes.byref(size),
                                                  ctypes.byref(count), buffer)
        if status != ERROR_SUCCESS:
            return None
        items = ctypes.cast(buffer, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W))
        return sum(items[i].FmtValue.doubleValue for i in range(count.value)) / 2**20

    def close(self):
        pdh.PdhCloseQuery(self.query)


class GpuMemorySampler(threading.Thread):
    """Samples a process's shared GPU memory in the background; keeps the peak and the last value."""

    def __init__(self, pid, interval_s=1.0):
        super().__init__(daemon=True)
        self.reader = GpuMemoryReader(pid)
        self.interval_s = interval_s
        self.peak_mib = None
        self.last_mib = None
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            value = self.reader.read_mib()
            if value is not None:
                self.last_mib = value
                self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._halt.wait(self.interval_s)

    def stop(self):
        self._halt.set()
        self.join()
        self.reader.close()
        return self.peak_mib


if __name__ == "__main__":
    import sys

    reader = GpuMemoryReader(int(sys.argv[1]))
    reader.read_mib()
    print(f"{reader.read_mib():.1f} MiB shared GPU memory")
