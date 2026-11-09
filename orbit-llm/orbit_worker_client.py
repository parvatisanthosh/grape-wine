"""Python side of cpp/orbit_worker.exe: a long-lived process that keeps models loaded.

orbit_exec.run_configuration starts a fresh orbit_run.exe for every run, so every
run reloads the model. OrbitWorker starts orbit_worker.exe once and talks to it
through pipes: one command line in, one JSON line back.

    with OrbitWorker() as worker:
        worker.load("INT4@gpu", "C:/Users/parva/tinyllama-int4",
                    label="INT4", device="GPU", cache_dir="ov_cache/gpu")
        reply = worker.generate("INT4@gpu", prompt_tokens=1024, output_tokens=64)
        print(reply["ttft_ms"])
        worker.unload("INT4@gpu")

Every method returns the worker's reply as a dict. A command the worker rejects
comes back with reply["status"] == "error" and an "error" message; it does not
raise. WorkerError is raised only when the worker itself stops responding.
"""

import json
import os
import subprocess
from pathlib import Path

from orbit_exec import OPENVINO_LIBS, OPTION_FLAGS


ORBIT_WORKER = Path(__file__).parent / "cpp" / "build" / "Release" / "orbit_worker.exe"

# Configuration keys that make sense when loading (the rest describe a request).
LOAD_KEYS = {
    "label",
    "device",
    "backend",
    "prefix_caching",
    "max_batched_tokens",
    "cache_size_gb",
    "kv_blocks",
    "kv_precision",
    "threads",
    "cores",
    "cache_dir",
}


class WorkerError(RuntimeError):
    """The worker exited, or answered out of step with the commands sent."""


def quote(word):
    """Quote a word for the worker's command line if it contains spaces."""
    word = str(word)
    if '"' in word or "\n" in word:
        raise ValueError(f"Cannot send {word!r}: quotes and newlines are not allowed")
    return f'"{word}"' if (" " in word or "\t" in word or word == "") else word


class OrbitWorker:
    def __init__(self, command=None):
        """Start the worker. command overrides the executable (used by tests)."""
        if command is None:
            if not ORBIT_WORKER.exists():
                raise FileNotFoundError(
                    f"orbit_worker.exe not found: {ORBIT_WORKER}. "
                    "Build it first: cmake --build cpp/build --config Release"
                )
            command = [str(ORBIT_WORKER)]

        environment = os.environ.copy()
        environment["PATH"] = str(OPENVINO_LIBS) + os.pathsep + environment.get("PATH", "")

        # stdin/stdout are pipes for the protocol; stderr (progress messages)
        # goes straight to the console, so it can never fill a pipe and block.
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )

        ready = self._read()
        if ready.get("command") != "ready":
            self.process.kill()
            raise WorkerError(f"Expected a ready line from the worker, got {ready}")
        self.ready = ready

    # -----------------------------------------------------------------------
    # Protocol

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            code = self.process.wait()
            raise WorkerError(f"orbit_worker exited (code {code}) without replying")
        return json.loads(line)

    def _send(self, *words):
        if self.process.poll() is not None:
            raise WorkerError(f"orbit_worker is not running (code {self.process.returncode})")

        self.process.stdin.write(" ".join(quote(word) for word in words) + "\n")
        self.process.stdin.flush()

        reply = self._read()
        # The worker answers every command exactly once, in order. A reply for a
        # different command means the two sides are out of step.
        if reply.get("command") != words[0]:
            raise WorkerError(f"Sent {words[0]!r} but the worker replied to {reply.get('command')!r}")
        return reply

    # -----------------------------------------------------------------------
    # Commands

    def load(self, slot, model, **configuration):
        flags = []
        for key, value in configuration.items():
            if key not in LOAD_KEYS:
                raise ValueError(f"{key!r} is not a load option (allowed: {sorted(LOAD_KEYS)})")
            flags += [OPTION_FLAGS[key], value]
        return self._send("load", slot, "--model", model, *flags)

    def generate(self, slot, prompt_tokens, output_tokens, prompt_id="run_1"):
        return self._send(
            "generate",
            slot,
            "--prompt-tokens",
            prompt_tokens,
            "--output-tokens",
            output_tokens,
            "--prompt-id",
            prompt_id,
        )

    def unload(self, slot):
        return self._send("unload", slot)

    def status(self):
        return self._send("status")

    def close(self):
        """Ask the worker to quit; kill it if it does not exit in time."""
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=30)
            except (WorkerError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()