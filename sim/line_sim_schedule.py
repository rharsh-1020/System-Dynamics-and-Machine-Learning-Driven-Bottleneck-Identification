import os, json, statistics as stats
from pathlib import Path
import simpy, pandas as pd, numpy as np

DEFAULTS = {
    "sim_time_min": 480,
    "interarrival_min": 0.8,
    "stations": [
        {"name": "S1", "rate_ppm": 1.0},
        {"name": "S2", "rate_ppm": 0.6},
        {"name": "S3", "rate_ppm": 1.0},
    ],
}

def load_params():
    cfg = DEFAULTS.copy()
    yml = Path(__file__).with_name("params.yaml")
    if yml.exists():
        try:
            import yaml
            with open(yml, "r") as f:
                user = yaml.safe_load(f) or {}
            for k,v in user.items(): cfg[k]=v
            print(f"[INFO] Loaded {yml}")
        except Exception as e:
            print(f"[WARN] YAML parse failed: {e}")
    return cfg

class Station:
    def __init__(self, env, name, rate_ppm):
        self.env = env; self.name=name; self.rate=rate_ppm
        self.res = simpy.Resource(env, capacity=1)
        self.busy_time=0.0; self.last_start=None; self.completed=0
        self.queue_ts=[]; self.time_ts=[]
    def ptime(self): return 1.0/self.rate if self.rate>0 else 1e9
    def start(self): self.last_start=self.env.now
    def end(self):
        if self.last_start is not None:
            self.busy_time += self.env.now - self.last_start
            self.last_start=None
        self.completed += 1

def track(env, stations, every=1.0):
    while True:
        for s in stations:
            s.queue_ts.append(len(s.res.queue)); s.time_ts.append(env.now)
        yield env.timeout(every)

def process_part(env, stations):
    for st in stations:
        with st.res.request() as req:
            yield req; st.start(); yield env.timeout(st.ptime()); st.end()

def source(env, stations, interarrival):
    while True:
        env.process(process_part(env, stations)); yield env.timeout(interarrival)

def schedule_events(env, stations_by_name, events_df):
    # events_df columns: at_min, station, new_rate_ppm
    for _,row in events_df.sort_values("at_min").iterrows():
        t = float(row["at_min"]); st = str(row["station"]); new = float(row["new_rate_ppm"])
        def changer(t=t, st=st, new=new):
            yield env.timeout(t)
            if st in stations_by_name:
                stations_by_name[st].rate = new
        env.process(changer())

def run(tag="baseline", events_csv=None):
    cfg = load_params()
    env = simpy.Environment()
    stations = [Station(env, s["name"], s["rate_ppm"]) for s in cfg["stations"]]
    byname = {s.name:s for s in stations}
    # default mid-run shift for S2/S3 to demonstrate shifting bottleneck
    # If you want to include it along with schedule, add those rows to the CSV.
    env.process(source(env, stations, interarrival=cfg["interarrival_min"]))
    env.process(track(env, stations, every=1.0))
    if events_csv:
        df = pd.read_csv(events_csv)
        schedule_events(env, byname, df)
        print(f"[INFO] Loaded schedule {events_csv} with {len(df)} events")
    env.run(until=cfg["sim_time_min"])
    # metrics
    def util(st): return 100.0*st.busy_time/cfg["sim_time_min"]
    metrics = {s.name:{"util_percent":round(util(s),2),
                       "avg_queue":round(np.mean(s.queue_ts),3) if s.queue_ts else 0.0,
                       "completed":int(s.completed)} for s in stations}
    throughput = stations[-1].completed
    # save
    out_root = Path(__file__).resolve().parents[1]
    figs = out_root/"figs"; sims = out_root/"sim"
    figs.mkdir(exist_ok=True, parents=True); sims.mkdir(exist_ok=True, parents=True)
    met_df = pd.DataFrame(metrics).T; met_df["station"]=met_df.index
    met_df.to_csv(sims/f"sim_metrics_{tag}.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # bars
    plt.figure(figsize=(5,3)); plt.bar(met_df["station"], met_df["util_percent"])
    plt.ylabel("Utilization (%)"); plt.title(f"Station Utilization — {tag}"); plt.tight_layout()
    plt.savefig(figs/f"sim_util_bar_{tag}.png", dpi=150); plt.close()
    plt.figure(figsize=(5,3)); plt.bar(met_df["station"], met_df["avg_queue"])
    plt.ylabel("Average Queue"); plt.title(f"Average Queue — {tag}"); plt.tight_layout()
    plt.savefig(figs/f"sim_queue_bar_{tag}.png", dpi=150); plt.close()
    # queues over time
    ts = pd.DataFrame({"time_min": stations[0].time_ts})
    for s in stations: ts[f"{s.name}_queue"]=s.queue_ts
    ts.to_csv(sims/f"queue_ts_{tag}.csv", index=False)
    plt.figure(figsize=(7,3))
    for s in stations: plt.plot(ts["time_min"], ts[f"{s.name}_queue"], label=s.name)
    plt.xlabel("Time (min)"); plt.ylabel("Queue"); plt.title(f"Queues over time — {tag}")
    plt.legend(ncol=3); plt.tight_layout(); plt.savefig(figs/f"sim_queue_ts_{tag}.png", dpi=150); plt.close()
    # summary
    with open(sims/f"sim_summary_{tag}.json","w") as f:
        json.dump({"throughput":int(throughput),"metrics":metrics}, f, indent=2)
    print(f"=== {tag.upper()} === Throughput: {throughput}  | metrics saved to sim/ & figs/")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--events", default=None, help="CSV with at_min,station,new_rate_ppm")
    args=ap.parse_args()
    run(tag=args.tag, events_csv=args.events)
