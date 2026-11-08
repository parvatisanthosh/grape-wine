"""A stand-in for orbit_worker.exe that speaks the same protocol, for tests.

It loads nothing: "generate" answers instantly. A repeated prompt id on the
same slot is reported as a fast prefix-cache hit, like the real worker would.
Memory is faked as 400 MiB for the runtime plus 500 MiB per loaded slot, so
tests can see that keeping models loaded costs memory.
"""

RUNTIME_MIB = 400.0
SLOT_MIB = 500.0

import json
import shlex
import sys


def reply(**fields):
    print(json.dumps(fields), flush=True)


def main():
    slots = {}
    reply(command="ready", slot="", status="success")

    for line in sys.stdin:
        words = shlex.split(line)
        if not words:
            continue
        command, rest = words[0], words[1:]
        slot = rest[0] if rest else ""

        if command == "quit":
            reply(command="quit", slot="", status="success")
            return
        if command == "crash":
            sys.exit(3)  # exits without replying, to test error handling
        if command == "load":
            if slot in slots:
                reply(command="load", slot=slot, status="error",
                      error=f"slot '{slot}' is already loaded; unload it first")
                continue
            flags = dict(zip(rest[1::2], rest[2::2]))
            slots[slot] = {"seen": set(), "count": 0, "device": flags.get("--device", "CPU")}
            reply(command="load", slot=slot, status="success", pipeline_construction_ms=2500.0,
                  device=slots[slot]["device"], loaded_slots=",".join(slots),
                  rss_after_mib=RUNTIME_MIB + SLOT_MIB * len(slots))
        elif command == "generate":
            if slot not in slots:
                reply(command="generate", slot=slot, status="error", error=f"slot '{slot}' is not loaded")
                continue
            flags = dict(zip(rest[1::2], rest[2::2]))
            state = slots[slot]
            prompt_id = flags.get("--prompt-id", "run_1")
            hit = prompt_id in state["seen"]
            state["seen"].add(prompt_id)
            state["count"] += 1
            reply(command="generate", slot=slot, status="success", prompt_id=prompt_id,
                  generation_index=state["count"],
                  requested_input_tokens=int(flags["--prompt-tokens"]),
                  requested_output_tokens=int(flags["--output-tokens"]),
                  ttft_ms=20.0 if hit else 1500.0, tpot_ms_per_token=28.0,
                  generation_ms=(20.0 if hit else 1500.0) + 28.0 * int(flags["--output-tokens"]),
                  device=state["device"], loaded_slots=",".join(slots),
                  sampled_peak_rss_mib=RUNTIME_MIB + SLOT_MIB * len(slots))
        elif command == "unload":
            if slot not in slots:
                reply(command="unload", slot=slot, status="error", error=f"slot '{slot}' is not loaded")
                continue
            served = slots.pop(slot)["count"]
            reply(command="unload", slot=slot, status="success", unload_ms=150.0,
                  generations_served=served, freed_commit_mib=900.0, loaded_slots=",".join(slots))
        elif command == "status":
            reply(command="status", slot="", status="success", loaded_slots=",".join(slots))
        else:
            reply(command=command, slot=slot, status="error", error=f"Unknown command '{command}'")


if __name__ == "__main__":
    main()