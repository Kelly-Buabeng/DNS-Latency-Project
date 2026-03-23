# DNS Latency and Its Impact on Web Page Load Time
## Project README & Execution Guide

---

## Project Structure

```
dns-latency-project/
├── dns_measure.py        Phase 1 & 2 — Data collection (DNS + TCP + HTTP timing)
├── analysis.py           Phase 3     — Statistical analysis & plots
├── queuing_model.py      Phase 4 & 5 — M/M/1 model + discrete-event simulation
├── results/
│   ├── raw_measurements.csv      One row per (trial, resolver, hostname)
│   ├── summary_stats.csv         Descriptive stats per (resolver, hostname)
│   └── distribution_fits.csv    KS test results for Exp/Normal/LogNormal fits
└── plots/
    ├── dns_boxplot.png           DNS latency distribution per resolver
    ├── dns_cdf.png               ECDF comparison across resolvers
    ├── dns_mean_bar.png          Mean DNS latency with 95% CI error bars
    ├── dns_heatmap.png           Median DNS latency (resolver × hostname)
    ├── tcp_boxplot.png           TCP handshake latency per resolver
    ├── component_stacked.png     DNS + TCP stacked bar (delay model)
    ├── rho_sweep.png             M/M/1 sojourn time vs utilisation
    ├── sim_vs_analytical.png     Simulation vs analytical comparison
    └── sojourn_distribution.png  Simulated histogram + ECDF vs theory
```

---

## Setup

### 1. Install dependencies

```bash
pip install dnspython requests numpy pandas matplotlib scipy seaborn
```

### 2. Find your DNS resolver IPs

**ISP resolver:**
```bash
cat /etc/resolv.conf        # Linux
# or
nmcli dev show | grep DNS   # Linux (NetworkManager)
```

**Campus DNS:**
- Check with your university IT department, or
- Run `dig @8.8.8.8 your.campus.hostname NS` and look at authority records

### 3. Edit dns_measure.py

Open `dns_measure.py` and update:
```python
RESOLVERS = {
    "google_public":     "8.8.8.8",
    "cloudflare_public": "1.1.1.1",
    "isp":               "YOUR_ISP_DNS_IP",    # ← change this
    "campus":            "YOUR_CAMPUS_DNS_IP", # ← change this
}
```

---

## Running the Project

### Step 1 — Collect measurements

```bash
python dns_measure.py
```

This runs 30 trials × 4 resolvers × 8 hostnames = **960 measurements**.
Estimated runtime: ~12–15 minutes.

### Step 2 — Statistical analysis

```bash
python analysis.py
```

Reads `results/raw_measurements.csv`, produces all statistical plots
and the distribution fitting CSV.

### Step 3 — Queuing model + simulation

```bash
python queuing_model.py --csv results/summary_stats.csv
```

Uses empirical mean DNS times to parameterise the M/M/1 model,
runs discrete-event simulation, produces comparison plots.

---

## Core Delay Model

```
T_total = T_dns + T_tcp + T_tls (optional) + T_transfer
```

| Component       | What it measures                        | How measured              |
|-----------------|----------------------------------------|---------------------------|
| T_dns           | DNS resolver lookup time               | dnspython directed query  |
| T_tcp           | TCP three-way handshake time           | socket.connect() timing   |
| T_tls           | TLS handshake (HTTPS only)             | Not measured (HTTP only)  |
| T_transfer      | Time to first byte of HTTP response    | requests streaming GET    |

---

## Queuing Model Summary

Each DNS resolver is modelled as an **M/M/1 queue**:
- **M**: Poisson arrivals (memoryless inter-arrival times)
- **M**: Exponential service times (one exponential server)
- **1**: Single server

Key formula — **Little's Law**:
```
L = λ · W
```
where:
- L = mean number of queries in the system
- λ = arrival rate (queries/second)
- W = mean sojourn time (time in system)

**M/M/1 sojourn time**:
```
W = 1 / (μ - λ)    [in seconds]
```
where μ = 1 / mean_service_time = service rate.

---

## Course Outcomes Addressed

| Outcome | How this project addresses it |
|---------|------------------------------|
| 1. Network performance metrics | DNS lookup time, TCP handshake time, page load estimate |
| 2. Measurement methodology | Active measurement; end-to-end (DNS→TCP→HTTP); 30 repeated trials |
| 3. Traffic characterisation | Distribution fitting (Exp/Normal/LogNorm), CoV, burstiness analysis |
| 4. Little's Law & queuing | M/M/1 model per resolver, ρ sweep, queue length estimation |
| 5. Simulation | Custom discrete-event simulator, M/M/1 validation |
| 6. Analytical vs simulation | Side-by-side W comparison table and bar chart |
| 7. Bottleneck diagnosis | Identify slowest resolver; quantify impact on page load |
| 8. Technical communication | (Report template in report_outline.md) |
