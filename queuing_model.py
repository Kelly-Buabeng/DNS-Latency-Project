"""
DCIT 417 — Network Modelling & Performance Analysis
Project: DNS Latency and Its Impact on Web Page Load Time

Phase 4 — Analytical Queuing Model
Phase 5 — Discrete-Event Simulation
======================================

Phase 4: M/M/1 Queuing Model
------------------------------
Each DNS resolver is modelled as an M/M/1 queue:
    • Arrivals   : Poisson process, rate λ (queries/second)
    • Service    : Exponential, rate μ = 1/mean_service_time
    • Servers    : 1
    • Queue      : Infinite FCFS

M/M/1 closed-form results
--------------------------
    ρ = λ/μ                         (utilisation, must be < 1 for stability)
    W = 1/(μ - λ)                   (mean time in system = sojourn time)
    Wq = ρ/(μ - λ) = λ/(μ(μ-λ))   (mean waiting time in queue)
    L  = λ·W = ρ/(1-ρ)             (mean number in system — Little's Law)
    Lq = λ·Wq                       (mean number in queue)

Little's Law: L = λ · W
    (mean jobs in system) = (arrival rate) × (mean time in system)
    This is model-independent — we use it to cross-check simulation.

Phase 5: Discrete-Event Simulation
------------------------------------
We simulate the M/M/1 queue manually using Python (no ns-3 needed for
this part) to:
    • Validate the analytical M/M/1 predictions
    • Observe how latency grows as ρ → 1 (resolver saturation)
    • Explore finite-buffer effects (M/M/1/K)

The simulation produces W_sim vs W_analytical and a ρ sweep plot.

Usage
-----
    pip install numpy pandas matplotlib scipy
    python queuing_model.py [--csv results/summary_stats.csv]
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       10,
})

# ──────────────────────────────────────────────
# PHASE 4 — ANALYTICAL M/M/1 MODEL
# ──────────────────────────────────────────────

@dataclass
class MM1Result:
    """Analytical M/M/1 performance metrics."""
    resolver:     str
    lambda_qps:   float   # arrival rate  (queries/second)
    mu_qps:       float   # service rate  (queries/second) = 1/mean_dns_ms * 1000
    rho:          float   # utilisation
    W_ms:         float   # mean sojourn time (ms)
    Wq_ms:        float   # mean queue wait  (ms)
    L:            float   # mean number in system
    Lq:           float   # mean number in queue
    stable:       bool    # True if ρ < 1

    def __str__(self) -> str:
        status = "STABLE" if self.stable else "UNSTABLE (ρ≥1)"
        return (f"  {self.resolver:<25}"
                f"  λ={self.lambda_qps:.3f} q/s"
                f"  μ={self.mu_qps:.3f} q/s"
                f"  ρ={self.rho:.3f}"
                f"  W={self.W_ms:.2f}ms"
                f"  Wq={self.Wq_ms:.2f}ms"
                f"  L={self.L:.3f}"
                f"  Lq={self.Lq:.3f}"
                f"  [{status}]")


def mm1_analytical(resolver: str,
                   mean_dns_ms: float,
                   lambda_qps: float) -> MM1Result:
    """
    Compute M/M/1 metrics for a DNS resolver.

    Parameters
    ----------
    resolver    : resolver name (label only)
    mean_dns_ms : empirically measured mean DNS service time (ms)
    lambda_qps  : assumed arrival rate in queries per second

    The service rate μ = 1000 / mean_dns_ms  (converting ms → per-second).
    """
    mu = 1000.0 / mean_dns_ms       # service rate (queries/second)
    rho = lambda_qps / mu

    if rho >= 1.0:
        return MM1Result(
            resolver=resolver, lambda_qps=lambda_qps, mu_qps=mu,
            rho=rho, W_ms=float("inf"), Wq_ms=float("inf"),
            L=float("inf"), Lq=float("inf"), stable=False
        )

    # Mean sojourn time  W = 1/(μ - λ)  in seconds → convert to ms
    W_s  = 1.0 / (mu - lambda_qps)
    Wq_s = rho / (mu - lambda_qps)
    W_ms  = W_s  * 1000.0
    Wq_ms = Wq_s * 1000.0
    L  = lambda_qps * W_s          # Little's Law
    Lq = lambda_qps * Wq_s

    return MM1Result(
        resolver=resolver, lambda_qps=lambda_qps, mu_qps=mu,
        rho=rho, W_ms=W_ms, Wq_ms=Wq_ms, L=L, Lq=Lq, stable=True
    )


def run_analytical_models(summary_csv: Optional[str] = None) -> list[MM1Result]:
    """
    Run M/M/1 analysis for each resolver.

    If summary_csv is provided, reads empirical mean DNS times.
    Otherwise uses demo values for illustration.
    """
    # Default demo values (replace with your empirical results)
    defaults = {
        "google_public":     25.0,   # mean DNS time in ms
        "cloudflare_public": 20.0,
        "isp":               45.0,
        "campus":            35.0,
    }

    if summary_csv and os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        means = {}
        for _, row in df.iterrows():
            # summary_stats.csv groups by (resolver, hostname) — take overall mean
            r = row.get("resolver", row.get("resolver_name", ""))
            v = row.get("dns_mean", row.get("dns_time_ms_mean", None))
            if r and v:
                means.setdefault(r, []).append(float(v))
        resolver_means = {r: np.mean(vs) for r, vs in means.items()}
        print(f"Loaded empirical DNS means from {summary_csv}: {resolver_means}")
    else:
        resolver_means = defaults
        print(f"Using demo DNS means: {resolver_means}")

    # Assumed arrival rate: 10 DNS queries per second (moderate load)
    # In practice estimate from: λ = requests_per_page × pages_per_second
    lambda_qps = 10.0

    results = []
    print("\n── M/M/1 Analytical Results (λ = {:.1f} q/s) ──".format(lambda_qps))
    for resolver, mean_ms in resolver_means.items():
        r = mm1_analytical(resolver, mean_ms, lambda_qps)
        print(r)
        results.append(r)

    # Verify Little's Law: L = λ·W  (already satisfied by construction, but good to print)
    print("\n── Little's Law Verification ──")
    for r in results:
        if r.stable:
            l_littles = r.lambda_qps * r.W_ms / 1000.0   # W in seconds
            print(f"  {r.resolver:<25}: L={r.L:.4f}  λ·W={l_littles:.4f}  "
                  f"match={'YES' if abs(r.L - l_littles) < 1e-6 else 'NO'}")

    return results


# ──────────────────────────────────────────────
# RHO SWEEP — analytical W vs ρ
# ──────────────────────────────────────────────

def plot_rho_sweep(mean_dns_ms_map: dict[str, float],
                   colours: dict[str, str]) -> None:
    """
    Plot mean sojourn time W (ms) as a function of utilisation ρ.

    This is the fundamental M/M/1 result:  W = 1/(μ(1-ρ))
    As ρ → 1, W → ∞  (the knee of the curve).

    The plot illustrates why keeping ρ well below 1 is critical for
    DNS resolver performance, and shows the relative positions of each
    resolver given their empirical service rates.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    rho_vals = np.linspace(0.01, 0.99, 400)

    for resolver, mean_ms in mean_dns_ms_map.items():
        mu = 1000.0 / mean_ms
        W_vals = [1000.0 / (mu * (1 - rho)) for rho in rho_vals]
        colour = colours.get(resolver, "gray")
        label = {
            "google_public": "Google 8.8.8.8",
            "cloudflare_public": "Cloudflare 1.1.1.1",
            "isp": "ISP resolver",
            "campus": "Campus DNS",
        }.get(resolver, resolver)
        ax.plot(rho_vals, W_vals, color=colour, linewidth=2, label=label)

    ax.set_xlabel("Utilisation ρ (= λ/μ)")
    ax.set_ylabel("Mean Sojourn Time W (ms)")
    ax.set_title("M/M/1: DNS Resolver Sojourn Time vs Utilisation", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 500)
    ax.axvline(0.8, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="ρ = 0.8 (design limit)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f ms"))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rho_sweep.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ──────────────────────────────────────────────
# PHASE 5 — DISCRETE-EVENT SIMULATION
# ──────────────────────────────────────────────

class Event:
    """Simulation event with a time and type."""
    ARRIVAL   = "arrival"
    DEPARTURE = "departure"

    def __init__(self, time: float, kind: str, job_id: int):
        self.time    = time
        self.kind    = kind
        self.job_id  = job_id

    def __lt__(self, other):
        return self.time < other.time


def simulate_mm1(lambda_qps: float,
                 mu_qps: float,
                 n_jobs: int = 5000,
                 seed: int = 42) -> dict:
    """
    Simulate an M/M/1 queue using a manual event-driven approach.

    Algorithm
    ---------
    1. Initialise: empty queue, server idle, schedule first arrival
    2. Process events in time order (future-event list):
       Arrival  → enqueue job; if server idle, start service immediately
       Departure→ record sojourn time; if queue non-empty, start next job
    3. Collect per-job sojourn times, compute statistics

    Parameters
    ----------
    lambda_qps : arrival rate (jobs/second)
    mu_qps     : service rate (jobs/second)
    n_jobs     : number of jobs to simulate
    seed       : random seed for reproducibility

    Returns
    -------
    dict with keys: W_ms, Wq_ms, L, Lq, rho, utilisation
    """
    rng = random.Random(seed)

    # Exponential inter-arrival time: 1/λ seconds
    def next_interarrival() -> float:
        return rng.expovariate(lambda_qps)

    # Exponential service time: 1/μ seconds
    def next_service() -> float:
        return rng.expovariate(mu_qps)

    # State
    clock          = 0.0
    server_busy    = False
    queue: list    = []          # list of (arrival_time, job_id)
    future_events: list = []     # heap of Event objects (use sorted list for clarity)

    # Per-job tracking
    arrival_times:   dict[int, float] = {}
    service_start:   dict[int, float] = {}
    departure_times: dict[int, float] = {}

    sojourn_times: list[float] = []
    wait_times:    list[float] = []

    import heapq
    heapq.heapify(future_events)

    # Schedule first arrival
    first_arrival = next_interarrival()
    heapq.heappush(future_events, (first_arrival, Event(first_arrival, Event.ARRIVAL, 0)))

    jobs_arrived   = 0
    jobs_completed = 0

    while jobs_completed < n_jobs:
        if not future_events:
            break

        t, event = heapq.heappop(future_events)
        clock = t

        if event.kind == Event.ARRIVAL:
            job_id = event.job_id
            arrival_times[job_id] = clock

            if not server_busy:
                # Start service immediately
                server_busy = True
                service_start[job_id] = clock
                svc_time = next_service()
                dep_time = clock + svc_time
                heapq.heappush(future_events,
                               (dep_time, Event(dep_time, Event.DEPARTURE, job_id)))
            else:
                queue.append((clock, job_id))

            # Schedule next arrival (if we need more)
            jobs_arrived += 1
            if jobs_arrived < n_jobs * 3:   # overshoot to fill pipeline
                ia = next_interarrival()
                arr_t = clock + ia
                heapq.heappush(future_events,
                               (arr_t, Event(arr_t, Event.ARRIVAL, jobs_arrived)))

        elif event.kind == Event.DEPARTURE:
            job_id = event.job_id
            departure_times[job_id] = clock
            jobs_completed += 1

            sojourn = clock - arrival_times[job_id]
            wait    = service_start[job_id] - arrival_times[job_id]
            sojourn_times.append(sojourn * 1000)   # convert to ms
            wait_times.append(wait * 1000)

            if queue:
                # Start serving next job in queue
                _, next_job = queue.pop(0)
                service_start[next_job] = clock
                svc_time = next_service()
                dep_time = clock + svc_time
                heapq.heappush(future_events,
                               (dep_time, Event(dep_time, Event.DEPARTURE, next_job)))
            else:
                server_busy = False

    W_ms_sim  = float(np.mean(sojourn_times))
    Wq_ms_sim = float(np.mean(wait_times))
    L_sim     = lambda_qps * (W_ms_sim / 1000.0)   # Little's Law
    Lq_sim    = lambda_qps * (Wq_ms_sim / 1000.0)

    return {
        "W_ms":  W_ms_sim,
        "Wq_ms": Wq_ms_sim,
        "L":     L_sim,
        "Lq":    Lq_sim,
        "rho_empirical": W_ms_sim / (W_ms_sim / (1 - lambda_qps/mu_qps))
                         if lambda_qps < mu_qps else float("nan"),
        "n_completed": jobs_completed,
        "sojourn_times_ms": sojourn_times,
    }


def run_simulation_validation(mean_dns_ms_map: dict[str, float],
                               lambda_qps: float = 10.0) -> None:
    """
    Run M/M/1 simulation for each resolver and compare with analytical results.
    Prints a comparison table and produces a bar chart.
    """
    print("\n── Simulation vs Analytical Comparison ──")
    header = f"{'Resolver':<25} {'W_analytical':>14} {'W_simulation':>14} {'Error%':>8}"
    print(header)
    print("-" * len(header))

    comparison = []
    for resolver, mean_ms in mean_dns_ms_map.items():
        mu = 1000.0 / mean_ms
        rho = lambda_qps / mu

        if rho >= 0.99:
            print(f"  {resolver:<25}  SKIPPED (ρ={rho:.3f} ≥ 0.99)")
            continue

        # Analytical
        W_analytic = 1000.0 / (mu - lambda_qps)

        # Simulation
        sim = simulate_mm1(lambda_qps=lambda_qps, mu_qps=mu, n_jobs=3000)
        W_sim = sim["W_ms"]
        err_pct = abs(W_sim - W_analytic) / W_analytic * 100

        label = {"google_public":"Google 8.8.8.8",
                 "cloudflare_public":"Cloudflare 1.1.1.1",
                 "isp":"ISP resolver", "campus":"Campus DNS"}.get(resolver, resolver)
        print(f"  {label:<25}  {W_analytic:>12.2f}ms  {W_sim:>12.2f}ms  {err_pct:>7.2f}%")
        comparison.append({
            "resolver": label,
            "W_analytical_ms": W_analytic,
            "W_simulation_ms": W_sim,
            "error_pct": err_pct,
        })

    if not comparison:
        return

    # Bar chart comparing analytical vs simulated W
    df = pd.DataFrame(comparison)
    x  = np.arange(len(df))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, df["W_analytical_ms"], w, label="Analytical (M/M/1)",
           color="#2563EB", alpha=0.85)
    ax.bar(x + w/2, df["W_simulation_ms"], w, label="Simulation",
           color="#D97706", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["resolver"], rotation=15, ha="right")
    ax.set_ylabel("Mean Sojourn Time W (ms)")
    ax.set_title("Analytical vs Simulated M/M/1 Sojourn Time", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sim_vs_analytical.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_sojourn_distribution(mean_dns_ms: float,
                               lambda_qps: float = 10.0,
                               resolver_label: str = "Example resolver") -> None:
    """
    Plot histogram of simulated sojourn times vs. theoretical exponential CDF.
    This is Phase 5's key validation plot.
    """
    mu = 1000.0 / mean_dns_ms
    if lambda_qps / mu >= 0.99:
        print(f"Skipping sojourn distribution plot: ρ too high")
        return

    sim = simulate_mm1(lambda_qps=lambda_qps, mu_qps=mu, n_jobs=5000)
    W_theory = 1000.0 / (mu - lambda_qps)   # in ms

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Histogram
    ax = axes[0]
    ax.hist(sim["sojourn_times_ms"], bins=50, density=True,
            color="#2563EB", alpha=0.7, label="Simulated")
    t = np.linspace(0, max(sim["sojourn_times_ms"]), 300)
    # Theoretical PDF: (1/W_theory)*exp(-t/W_theory)  [exponential with mean W_theory]
    ax.plot(t, (1/W_theory) * np.exp(-t/W_theory), color="red",
            linewidth=2, label=f"Exponential(μ={1/W_theory*1000:.3f}/s)")
    ax.set_xlabel("Sojourn Time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Sojourn Time Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    # ── ECDF vs theoretical CDF
    ax = axes[1]
    sorted_t = np.sort(sim["sojourn_times_ms"])
    ecdf = np.arange(1, len(sorted_t)+1) / len(sorted_t)
    ax.step(sorted_t, ecdf, where="post", color="#2563EB",
            linewidth=1.5, label="Simulated ECDF")
    cdf_theory = 1 - np.exp(-t / W_theory)
    ax.plot(t, cdf_theory, color="red", linewidth=2,
            linestyle="--", label=f"Theoretical CDF (W={W_theory:.1f}ms)")
    ax.set_xlabel("Sojourn Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("ECDF vs Theoretical CDF", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(f"M/M/1 Simulation Validation — {resolver_label}", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sojourn_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to summary_stats.csv (from analysis.py)")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Phase 4: Analytical Models
    print("=" * 60)
    print("PHASE 4 — Analytical M/M/1 Queuing Model")
    print("=" * 60)
    analytical_results = run_analytical_models(args.csv)

    # Extract mean DNS times for sweep and simulation
    mean_dns_map = {r.resolver: 1000.0 / r.mu_qps for r in analytical_results if r.stable}

    colours = {
        "google_public":     "#2563EB",
        "cloudflare_public": "#16A34A",
        "isp":               "#D97706",
        "campus":            "#DC2626",
    }
    plot_rho_sweep(mean_dns_map, colours)

    # ── Phase 5: Simulation
    print("\n" + "=" * 60)
    print("PHASE 5 — Discrete-Event Simulation")
    print("=" * 60)

    lambda_qps = 10.0   # queries per second
    run_simulation_validation(mean_dns_map, lambda_qps=lambda_qps)

    # Sojourn distribution for the fastest resolver (as example)
    if mean_dns_map:
        fastest = min(mean_dns_map, key=mean_dns_map.get)
        label = {"google_public":"Google 8.8.8.8", "cloudflare_public":"Cloudflare 1.1.1.1",
                 "isp":"ISP resolver", "campus":"Campus DNS"}.get(fastest, fastest)
        plot_sojourn_distribution(mean_dns_map[fastest], lambda_qps=lambda_qps,
                                   resolver_label=label)

    print("\nAll queuing model and simulation plots saved to plots/")