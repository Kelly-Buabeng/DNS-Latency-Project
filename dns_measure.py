"""
DCIT 417 — Network Modelling & Performance Analysis
Project: DNS Latency and Its Impact on Web Page Load Time

dns_measure.py  —  Data Collection (Phase 1 & 2)
=================================================
Automatically detects the current network (SSID + DNS servers)
and names the resolver after the connected Wi-Fi network.

No manual IP configuration needed — just connect to the network
you want to measure, then run:

    python dns_measure.py

When you switch networks (e.g. campus → home → mobile hotspot),
re-run the script and the output CSV will use the SSID as the
resolver label automatically.

Outputs (SSID-named so multiple networks never overwrite each other)
--------------------------------------------------------------------
    results/raw_measurements_<ssid>.csv
    results/summary_stats_<ssid>.csv

To merge all networks into one file after collecting on all networks:
    python dns_measure.py --merge
"""

import csv
import os
import re
import socket
import statistics
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional

import dns.resolver          # pip install dnspython
import requests              # pip install requests

from network_detect import detect_network, build_resolver_config

# ──────────────────────────────────────────────
# CONFIGURATION — only edit these if needed
# ──────────────────────────────────────────────

TARGET_HOSTS = [
    "www.google.com",
    "www.youtube.com",
    "www.facebook.com",
    "www.wikipedia.org",
    "www.github.com",
    "www.amazon.com",
    "www.twitter.com",
    "www.reddit.com",
]

N_TRIALS          = 30      # repetitions per (resolver × host)
TCP_PORT          = 80
TCP_TIMEOUT_S     = 5
HTTP_TIMEOUT_S    = 10
INTER_TRIAL_DELAY = 0.2     # seconds between trials

OUTPUT_DIR = "results"


# ──────────────────────────────────────────────
# DATA STRUCTURE
# ──────────────────────────────────────────────

@dataclass
class Measurement:
    timestamp:      str
    trial:          int
    resolver_name:  str       # SSID-derived label, e.g. "ug_wifi"
    resolver_ip:    str
    ssid:           str       # raw SSID, e.g. "UG-WiFi"
    hostname:       str
    dns_time_ms:    Optional[float]
    tcp_time_ms:    Optional[float]
    http_time_ms:   Optional[float]
    total_time_ms:  Optional[float]
    dns_error:      str
    tcp_error:      str
    http_error:     str
    resolved_ip:    str


# ──────────────────────────────────────────────
# MEASUREMENT FUNCTIONS
# ──────────────────────────────────────────────

def measure_dns(hostname: str, resolver_ip: str) -> tuple[Optional[float], str, str]:
    """
    Directed DNS query to a specific resolver.
    Returns (elapsed_ms, resolved_ip, error_str).
    """
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = [resolver_ip]
    resolver.timeout  = 5
    resolver.lifetime = 5
    t_start = time.perf_counter()
    try:
        answers = resolver.resolve(hostname, "A")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return elapsed_ms, str(answers[0]), ""
    except dns.resolver.NXDOMAIN:
        return None, "", "NXDOMAIN"
    except dns.resolver.Timeout:
        return None, "", "Timeout"
    except dns.resolver.NoNameservers:
        return None, "", "NoNameservers"
    except Exception as exc:
        return None, "", str(exc)


def measure_tcp_handshake(ip: str, port: int = 80) -> tuple[Optional[float], str]:
    """
    Time the TCP three-way handshake (SYN → SYN-ACK → ACK) via socket.connect().
    Returns (elapsed_ms, error_str).
    """
    if not ip:
        return None, "No IP"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TCP_TIMEOUT_S)
        t_start = time.perf_counter()
        sock.connect((ip, port))
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        sock.close()
        return elapsed_ms, ""
    except socket.timeout:
        return None, "TCP timeout"
    except ConnectionRefusedError:
        return None, "Connection refused"
    except Exception as exc:
        return None, str(exc)


def measure_http_load(hostname: str, resolved_ip: str) -> tuple[Optional[float], str]:
    """
    Time from HTTP GET request to first byte received (TTFB).
    Returns (elapsed_ms, error_str).
    """
    if not resolved_ip:
        return None, "No IP"
    try:
        t_start = time.perf_counter()
        with requests.get(
            f"http://{hostname}/",
            headers={"Host": hostname},
            timeout=HTTP_TIMEOUT_S,
            stream=True,
            allow_redirects=True
        ) as resp:
            next(resp.iter_content(chunk_size=1024), None)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return elapsed_ms, ""
    except requests.exceptions.Timeout:
        return None, "HTTP timeout"
    except requests.exceptions.ConnectionError as exc:
        return None, f"Connection error: {exc}"
    except Exception as exc:
        return None, str(exc)


# ──────────────────────────────────────────────
# CORE MEASUREMENT LOOP
# ──────────────────────────────────────────────

def run_measurements(resolvers: dict[str, str], ssid: str) -> list[Measurement]:
    """
    Run all (resolver × hostname × trial) combinations.

    Parameters
    ----------
    resolvers : {label: ip} dict built from auto-detection
    ssid      : human-readable SSID recorded in every CSV row
    """
    results: list[Measurement] = []
    total = len(resolvers) * len(TARGET_HOSTS) * N_TRIALS
    count = 0

    for resolver_name, resolver_ip in resolvers.items():
        print(f"\n{'='*62}")
        print(f"Resolver : {resolver_name}  ({resolver_ip})")
        print(f"Network  : {ssid}")
        print(f"{'='*62}")

        for hostname in TARGET_HOSTS:
            print(f"  Host: {hostname}")

            for trial in range(1, N_TRIALS + 1):
                count += 1
                pct = count / total * 100
                print(f"    Trial {trial:2d}/{N_TRIALS}  [{pct:5.1f}%]", end="  ")

                # ① DNS timing
                dns_ms, resolved_ip, dns_err = measure_dns(hostname, resolver_ip)

                # ② TCP handshake
                tcp_ms, tcp_err = (
                    measure_tcp_handshake(resolved_ip, TCP_PORT)
                    if resolved_ip else (None, "Skipped (no IP)")
                )

                # ③ HTTP TTFB
                http_ms, http_err = (
                    measure_http_load(hostname, resolved_ip)
                    if resolved_ip else (None, "Skipped (no IP)")
                )

                # ④ Component-wise total
                components = [v for v in [dns_ms, tcp_ms] if v is not None]
                total_ms = sum(components) if components else None

                # Status line
                dns_s  = f"{dns_ms:7.1f}ms"  if dns_ms  else f"FAIL({dns_err})"
                tcp_s  = f"{tcp_ms:7.1f}ms"  if tcp_ms  else f"FAIL({tcp_err})"
                http_s = f"{http_ms:8.1f}ms" if http_ms else f"FAIL({http_err})"
                print(f"DNS={dns_s}  TCP={tcp_s}  HTTP={http_s}")

                results.append(Measurement(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    trial=trial,
                    resolver_name=resolver_name,
                    resolver_ip=resolver_ip,
                    ssid=ssid,
                    hostname=hostname,
                    dns_time_ms=round(dns_ms, 3) if dns_ms else None,
                    tcp_time_ms=round(tcp_ms, 3) if tcp_ms else None,
                    http_time_ms=round(http_ms, 3) if http_ms else None,
                    total_time_ms=round(total_ms, 3) if total_ms else None,
                    dns_error=dns_err,
                    tcp_error=tcp_err,
                    http_error=http_err,
                    resolved_ip=resolved_ip,
                ))

                time.sleep(INTER_TRIAL_DELAY)

    return results


# ──────────────────────────────────────────────
# CSV OUTPUT
# ──────────────────────────────────────────────

def save_raw(results: list[Measurement], path: str) -> None:
    """Write one row per measurement."""
    fieldnames = [f.name for f in fields(Measurement)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in results:
            writer.writerow({f.name: getattr(m, f.name) for f in fields(m)})
    print(f"\nRaw data  → {path}  ({len(results)} rows)")


def save_summary(results: list[Measurement], path: str) -> None:
    """Compute per-(resolver, hostname) descriptive statistics."""

    def pct(data: list[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        k = (len(s) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    def stats_for(values: list[float]) -> dict:
        if not values:
            return {k: None for k in ["n","mean","std","min","p25","median","p75","p95","max"]}
        return {
            "n":      len(values),
            "mean":   round(statistics.mean(values), 3),
            "std":    round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "min":    round(min(values), 3),
            "p25":    round(pct(values, 25), 3),
            "median": round(pct(values, 50), 3),
            "p75":    round(pct(values, 75), 3),
            "p95":    round(pct(values, 95), 3),
            "max":    round(max(values), 3),
        }

    groups: dict[tuple, list] = {}
    for m in results:
        groups.setdefault((m.ssid, m.resolver_name, m.hostname), []).append(m)

    rows = []
    for (ssid, resolver, host), measurements in sorted(groups.items()):
        dns_vals  = [m.dns_time_ms  for m in measurements if m.dns_time_ms  is not None]
        tcp_vals  = [m.tcp_time_ms  for m in measurements if m.tcp_time_ms  is not None]
        http_vals = [m.http_time_ms for m in measurements if m.http_time_ms is not None]
        row = {
            "ssid": ssid, "resolver": resolver, "hostname": host,
            "dns_success_rate":  len(dns_vals)  / len(measurements),
            "tcp_success_rate":  len(tcp_vals)  / len(measurements),
            "http_success_rate": len(http_vals) / len(measurements),
        }
        for prefix, vals in [("dns", dns_vals), ("tcp", tcp_vals), ("http", http_vals)]:
            for k, v in stats_for(vals).items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)

    if rows:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Summary   → {path}  ({len(rows)} groups)")


# ──────────────────────────────────────────────
# MULTI-NETWORK MERGE
# ──────────────────────────────────────────────

def merge_csv_files(output_dir: str = OUTPUT_DIR) -> None:
    """
    Merge all raw_measurements_*.csv files into one combined file.
    Run after collecting data on all three networks:
        python dns_measure.py --merge
    """
    import glob
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for merge: pip install pandas")
        return

    pattern = os.path.join(output_dir, "raw_measurements_*.csv")
    files = [f for f in glob.glob(pattern) if not f.endswith("_ALL.csv")]

    if not files:
        print(f"No files matching {pattern}")
        return

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    out = os.path.join(output_dir, "raw_measurements_ALL.csv")
    combined.to_csv(out, index=False)
    print(f"Merged {len(files)} network files → {out}  ({len(combined)} total rows)")
    for f in files:
        print(f"  {os.path.basename(f)}  ({len(pd.read_csv(f))} rows)")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── --merge flag ─────────────────────────────────────────────
    if "--merge" in sys.argv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        merge_csv_files()
        sys.exit(0)

    print("DCIT 417 — DNS Latency Measurement Tool")
    print("=" * 62)

    # ── Step 1: Auto-detect network ──────────────────────────────
    print("Detecting network environment...")
    net_info = detect_network(verbose=True)

    if not net_info.dns_servers:
        print("\nERROR: No DNS servers detected. Check your connection.")
        sys.exit(1)

    # ── Step 2: Build resolver config ────────────────────────────
    resolvers = build_resolver_config(net_info, always_include_public=True)
    ssid_label   = net_info.resolver_label()
    ssid_display = net_info.ssid or ssid_label

    print(f"\n── Resolver config (auto-built from SSID: '{ssid_display}') ──")
    for name, ip in resolvers.items():
        tag = "  ← this network's DNS" if name == ssid_label else ""
        print(f"  {name:<30} {ip}{tag}")

    # ── Step 3: Confirm ───────────────────────────────────────────
    total_m = len(resolvers) * len(TARGET_HOSTS) * N_TRIALS
    print(f"\nHosts      : {len(TARGET_HOSTS)} targets")
    print(f"Resolvers  : {len(resolvers)}")
    print(f"Trials     : {N_TRIALS} per (resolver × host)")
    print(f"Total      : {total_m} measurements  (~{total_m*INTER_TRIAL_DELAY/60:.1f} min)")

    try:
        if input("\nProceed? [Y/n] ").strip().lower() == "n":
            print("Aborted.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # ── Step 4: Run ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run_measurements(resolvers, ssid=ssid_display)

    # ── Step 5: Save (SSID-named files) ──────────────────────────
    safe_ssid    = re.sub(r'[^a-z0-9]+', '_', ssid_display.lower()).strip('_')
    raw_path     = os.path.join(OUTPUT_DIR, f"raw_measurements_{safe_ssid}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"summary_stats_{safe_ssid}.csv")

    save_raw(results, raw_path)
    save_summary(results, summary_path)

    print(f"\nNetwork label used : '{safe_ssid}'")
    print(f"To collect on another network, switch Wi-Fi and re-run.")
    print(f"To merge all networks : python dns_measure.py --merge")
    print(f"To analyse            : python analysis.py --csv {raw_path}")