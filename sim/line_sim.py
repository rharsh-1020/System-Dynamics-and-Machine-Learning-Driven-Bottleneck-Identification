"""
3-station serial line (S1 -> S2 -> S3) with a mid-run bottleneck shift.
- Tracks utilization, avg queue, completed parts (throughput).
- Reads optional sim/params.yaml; otherwise uses defaults.
- Saves figures to figs/ and CSVs to sim/.
"""

import os, math, json, statistics as stats
from pathlib import Path

import simpy
import pandas as pd
import numpy as np

# Try YAML params if available
DEFAULTS = {
    "sim_time_min": 480,
    "interarrival_min": 0.8,
    "stations": [
        {"name": "S1", "rate_ppm": 1.0},
        {"name": "S2", "rate_ppm": 0.6},  # bottleneck first half
        {"name": "S3", "rate_ppm": 1.0},
    ],
    "shift_event": {
        "at_min": 240,
        "changes": {
            "S2.rate_ppm": 1.0,  # fix S2
            "S3.rate_ppm": 0.5,  # make S3 new bottleneck
        },
    },
}

def load_params():
    cfg = DEFAULTS.copy()
    yml = Path(__file__).with_name("params.yaml")
    if yml.exists():
        try:
            import yaml
            with open(yml, "r") as f:
                user = yaml.safe_load(f) or {}
            # shallow merge
            for k, v in user.items():
                cfg[k] = v
            print(f"[INFO] Loaded params from {yml}")
        except Exception as e:
            print(f"[WARN] Could not parse {yml}: {e}\n[INFO] Using defaults.")
    else:
        print("[INFO] params.yaml not found; using defaults.")
    return cfg

class Station:
    def __init__(self, env, name, rate_ppm):
        self.env = env
        self.name = name
        self.rate = rate_ppm  # parts per minute
        self.res = simpy.Resource(env, capacity=1)
        self.busy_time = 0.0
        self.last_start = None
        self.completed = 0
        self.queue_ts = []   # queue length timeline (sampled)
        self.time_ts  = []   # matching timestamps

    def proc_time(self):
        # Deterministic service; for randomness use:
        # return random.expovariate(self.rate)
        return 1.0 / self.rate if self.rate > 0 else 1e9

    def start_service(self):
        self.last_start = self.env.now

    def end_service(self):
        if self.last_start is not None:
            self.busy_time += self.env.now - self.last_start
        self.completed += 1
        self.last_start = None

def track_queues(env, stations, sample_every=1.0):
    while True:
        for s in stations:
            s.queue_ts.append(len(s.res.queue))
            s.time_ts.append(env.now)
        yield env.timeout(sample_every)

def process_part(env, stations):
    for st in stations:
        with st.res.request() as req:
            yield req
            st.start_service()
            yield env.timeout(st.proc_time())
            st.end_service()

def source(env, stations, interarrival):
    while True:
        env.process(process_part(env, stations))
        yield env.timeout(interarrival)

def apply_changes(obj_lookup, changes: dict):
    """changes like {'S2.rate_ppm': 1.0, 'S3.rate_ppm': 0.5}"""
    for dotted, val in changes.items():
        top, attr = dotted.split(".", 1)
        if top in obj_lookup:
            target = obj_lookup[top]
            # only "rate_ppm" supported
            if attr == "rate_ppm":
                target.rate = float(val)

def run_scenario(cfg):
    env = simpy.Environment()
    # Stations
    stations = [Station(env, s["name"], s["rate_ppm"]) for s in cfg["stations"]]
    name_to_station = {s.name: s for s in stations}

    # Processes
    env.process(source(env, stations, interarrival=cfg["interarrival_min"]))
    env.process(track_queues(env, stations, sample_every=1.0))

    # Mid-run shift
    shift = cfg.get("shift_event", None)
    if shift:
        at = float(shift.get("at_min", cfg["sim_time_min"]/2))
        def shift_proc():
            yield env.timeout(at)
            apply_changes(name_to_station, shift.get("changes", {}))
        env.process(shift_proc())

    env.run(until=cfg["sim_time_min"])

    # Metrics
    def util(st): return 100.0 * st.busy_time / cfg["sim_time_min"]
    metrics = {
        st.name: {
            "util_percent": round(util(st), 2),
            "avg_queue": round(stats.mean(st.queue_ts), 3) if st.queue_ts else 0.0,
            "completed": int(st.completed),
        } for st in stations
    }

    # Throughput is completed at the last station
    throughput = stations[-1].completed

    # Time series DataFrame
    ts_rows = []
    for st in stations:
        ts_rows.append(pd.DataFrame({
            "time_min": st.time_ts,
            f"{st.name}_queue": st.queue_ts
        }))
    # Merge on time
    ts_df = ts_rows[0]
    for i in range(1, len(ts_rows)):
        ts_df = ts_df.merge(ts_rows[i], on="time_min", how="outer")
    ts_df = ts_df.sort_values("time_min").reset_index(drop=True)
    return metrics, throughput, ts_df, stations

def save_outputs(metrics, throughput, ts_df, outdir_figs="figs", outdir_sim="sim"):
    Path(outdir_figs).mkdir(parents=True, exist_ok=True)
    Path(outdir_sim).mkdir(parents=True, exist_ok=True)

    # Save metrics CSV
    met_df = pd.DataFrame(metrics).T
    met_df["station"] = met_df.index
    met_df.to_csv(Path(outdir_sim, "sim_metrics.csv"), index=False)

    # Save queue timeseries
    ts_df.to_csv(Path(outdir_sim, "queue_timeseries.csv"), index=False)

    # Plots (non-interactive)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Utilization bar
    plt.figure(figsize=(5,3))
    plt.bar(met_df["station"], met_df["util_percent"])
    plt.ylabel("Utilization (%)")
    plt.title("Station Utilization")
    plt.tight_layout()
    plt.savefig(Path(outdir_figs, "sim_utilization_bar.png"), dpi=150)
    plt.close()

    # Avg queue bar
    plt.figure(figsize=(5,3))
    plt.bar(met_df["station"], met_df["avg_queue"])
    plt.ylabel("Average Queue (parts)")
    plt.title("Average Queue Length")
    plt.tight_layout()
    plt.savefig(Path(outdir_figs, "sim_avg_queue_bar.png"), dpi=150)
    plt.close()

    # Queue over time
    qcols = [c for c in ts_df.columns if c.endswith("_queue")]
    plt.figure(figsize=(7,3))
    for c in qcols:
        plt.plot(ts_df["time_min"], ts_df[c], label=c.replace("_queue",""))
    plt.xlabel("Time (min)")
    plt.ylabel("Queue length")
    plt.title("Queues over time (bottleneck shift mid-run)")
    plt.legend(ncol=len(qcols))
    plt.tight_layout()
    plt.savefig(Path(outdir_figs, "sim_queue_over_time.png"), dpi=150)
    plt.close()

    # Summary txt
    summary = {
        "throughput_last_station": int(throughput),
        "metrics": metrics
    }
    with open(Path(outdir_sim, "sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def main():
    cfg = load_params()
    metrics, throughput, ts_df, stations = run_scenario(cfg)

    print("=== SIM RESULTS (baseline with mid-run shift) ===")
    print(f"Throughput (completed at last station): {throughput}")
    for st, m in metrics.items():
        print(f"{st}: util={m['util_percent']}%  avg_queue={m['avg_queue']}  completed={m['completed']}")

    save_outputs(metrics, throughput, ts_df)

if __name__ == "__main__":
    main()
