#!/usr/bin/env python3
"""Run paired, CPU/NUMA-bound PPO throughput cases on Linux.

Example: python3 tools/ppo_throughput.py --binary /path/tavernagent_v2 \
    --out /path/results --cases 8x1 16x1 32x1 64x1 --iterations 20
Each case is WORKERS x JOBS. This measures throughput, not policy strength.
Requires the Python standard library, lscpu, numactl and /usr/bin/time.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def topology():
    rows = subprocess.check_output(
        ["lscpu", "--parse=CPU,CORE,SOCKET,NODE"], text=True
    )
    nodes, seen = {}, set()
    allowed = os.sched_getaffinity(0)
    for line in rows.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, node = map(int, line.split(","))
        if cpu not in allowed or (socket, core) in seen:
            continue
        seen.add((socket, core))
        nodes.setdefault(node, []).append(cpu)
    return {node: sorted(cpus) for node, cpus in sorted(nodes.items())}


def placement(nodes, workers, jobs):
    """Spread jobs across NUMA nodes; never share physical cores within a case."""
    available = {node: list(cpus) for node, cpus in nodes.items()}
    result = []
    for job in range(jobs):
        node = list(available)[job % len(available)]
        cpus = available[node][:workers]
        if len(cpus) != workers:
            raise ValueError(f"case {workers}x{jobs} exceeds physical cores in node {node}")
        del available[node][:workers]
        result.append((node, cpus))
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_job(args, directory, workers, node, cpus):
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
        "numactl", "--physcpubind=" + ",".join(map(str, cpus)), f"--membind={node}",
        str(args.binary), "train", "--iterations", str(args.iterations),
        "--workers", str(workers), "--games", str(args.games), "--batch", str(args.batch),
        "--epochs", str(args.epochs), "--lr", str(args.lr), "--seed", str(args.seed),
        "--entropy-coef", str(args.entropy_coef),
        "--out", str(directory / "model.npz"),
    ]
    atomic_json(directory / "command.json", {"argv": command, "cpus": cpus, "node": node})
    started = time.monotonic()
    records = []
    previous = started
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"  # /usr/bin/time's field names are parsed below.
    with (directory / "stdout.log").open("w") as log, (directory / "stderr.log").open("w") as err:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True,
                                bufsize=1, env=environment)
        for line in proc.stdout:
            now = time.monotonic()
            log.write(line)
            log.flush()
            if line.startswith("{"):
                entry = json.loads(line)
                entry["received_monotonic"] = now
                entry["observed_iteration_seconds"] = now - previous
                previous = now
                records.append(entry)
        returncode = proc.wait()
    ended = time.monotonic()
    atomic_json(directory / "records.json", records)
    if returncode or len(records) != args.iterations:
        raise RuntimeError(f"{directory}: exit {returncode}, {len(records)} iterations; inspect logs")
    measured = records[args.warmup:]
    seconds = [r.get("iteration_seconds", r["observed_iteration_seconds"]) for r in measured]
    transitions = sum(r["transitions"] for r in measured)
    samples = sum(r["ppo"].get("trained_samples", r["ppo"]["updates"] * args.batch) for r in measured)
    resources = (directory / "resources.txt").read_text()
    rss = next(int(line.split(":")[-1]) for line in resources.splitlines() if "Maximum resident" in line)
    cpu_seconds = sum(float(line.split(":")[-1]) for line in resources.splitlines()
                      if "User time (seconds)" in line or "System time (seconds)" in line)
    summary = {
        "directory": str(directory), "workers": workers, "node": node, "cpus": cpus,
        "wall_seconds": ended - started, "started_monotonic": started, "ended_monotonic": ended,
        "all_transitions": sum(r["transitions"] for r in records),
        "measured_transitions": transitions, "measured_seconds": sum(seconds),
        "decisions_per_second": transitions / sum(seconds),
        "collect_decisions_per_second": transitions / sum(r["collect_seconds"] for r in measured),
        "trained_samples_per_second": samples / sum(r["train_seconds"] for r in measured),
        "trained_samples_exact": all("trained_samples" in r["ppo"] for r in measured),
        "iteration_seconds_median": statistics.median(seconds),
        "collect_seconds_mean": statistics.mean(r["collect_seconds"] for r in measured),
        "train_seconds_mean": statistics.mean(r["train_seconds"] for r in measured),
        "other_seconds_mean": statistics.mean(s-r["collect_seconds"]-r["train_seconds"] for s,r in zip(seconds, measured)),
        "updates": sum(r["ppo"]["updates"] for r in measured),
        "kl_stops": sum(r["ppo"]["early_stop"] for r in measured),
        "peak_rss_mib": rss / 1024,
        "cpu_seconds": cpu_seconds, "average_busy_cores": cpu_seconds / (ended - started),
        "measured_start_monotonic": measured[0]["received_monotonic"] - seconds[0],
        "measured_end_monotonic": measured[-1]["received_monotonic"],
    }
    atomic_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["8x1", "16x1", "32x1", "64x1"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=100_000_000)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.iterations or args.repeats < 1:
        parser.error("need iterations > warmup >= 0 and repeats >= 1")
    args.binary = args.binary.resolve(strict=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    nodes = topology()
    cases = [tuple(map(int, case.split("x"))) for case in args.cases]
    for workers, jobs in cases:
        if workers < 1 or jobs < 1:
            parser.error("workers and jobs must be positive")
        placement(nodes, workers, jobs)
    atomic_json(args.out / "manifest.json", {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(), "physical_cpus_by_node": nodes,
        "lscpu": subprocess.check_output(["lscpu"], text=True), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_note": "All cases/jobs use identical game and optimizer RNG seeds for controlled throughput comparison.",
    })
    results = []
    for repeat in range(args.repeats):
        for workers, jobs in (cases if repeat % 2 == 0 else list(reversed(cases))):
            name = f"w{workers}-j{jobs}-r{repeat}"
            print(f"start {name}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(run_job, args, args.out / name / f"job{i}", workers, node, cpus)
                           for i, (node, cpus) in enumerate(placement(nodes, workers, jobs))]
                summaries = [future.result() for future in futures]
            wall = max(s["ended_monotonic"] for s in summaries) - min(s["started_monotonic"] for s in summaries)
            measured_wall = max(s["measured_end_monotonic"] for s in summaries) - min(s["measured_start_monotonic"] for s in summaries)
            case = {"name": name, "workers": workers, "jobs": jobs, "repeat": repeat,
                    "aggregate_decisions_per_second": sum(s["all_transitions"] for s in summaries) / wall,
                    "warm_aggregate_decisions_per_second": sum(s["measured_transitions"] for s in summaries) / measured_wall,
                    "wall_seconds": wall, "job_summaries": summaries}
            results.append(case)
            atomic_json(args.out / "results.json", results)
            print(json.dumps({key: value for key, value in case.items() if key != "job_summaries"}), flush=True)


if __name__ == "__main__":
    main()
