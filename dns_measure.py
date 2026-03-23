"""
DCIT 417 — Network Modelling & Performance Analysis
Project: DNS Latency and Its Impact on Web Page Load Time
Author : [Your Name]
Date   : 2026

Phase 1 & 2 — Data Collection
==============================
Measures DNS lookup time, TCP handshake time, and estimated page load
time for a set of target hostnames, using multiple DNS resolvers:
    • Public  : Google (8.8.8.8), Cloudflare (1.1.1.1)
    • ISP     : your router's upstream resolver (auto-detected or set manually)
    • Campus  : your university's DNS server

Each measurement is repeated N times per (resolver, hostname) pair to
capture variability and build statistical distributions.

Outputs
-------
    results/raw_measurements.csv   — one row per (trial, resolver, host)
    results/summary_stats.csv      — mean, std, percentiles per group

Usage
-----
    pip install dnspython requests
    python dns_measure.py

Adjust RESOLVERS and TARGET_HOSTS below before running.
"""

import csv
import os
import socket
import statistics
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional

import dns.resolver          # pip install dnspython
import requests              # pip install requests

# ──────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ──────────────────────────────────────────────

RESOLVERS = {
    "google_public":     "8.8.8.8",
    "cloudflare_public": "1.1.1.1",
    "isp":               "192.168.1.1",   # ← your router/ISP DNS (check with: cat /etc/resolv.conf)
    "campus":            "10.0.0.1",      # ← your campus DNS IP (ask IT or check network settings)
}

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

N_TRIALS        = 30      # repetitions per (resolver, host) pair — minimum 30 for statistics
TCP_PORT        = 80      # port for TCP handshake measurement
TCP_TIMEOUT_S   = 5       # seconds before TCP attempt gives up
HTTP_TIMEOUT_S  = 10      # seconds before HTTP request gives up
INTER_TRIAL_DELAY = 0.2   # seconds between trials to avoid rate-limiting

OUTPUT_DIR = "results"

# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class Measurement:
    """One trial result for a single (resolver, hostname) pair."""
    timestamp:       str
    trial:           int
    resolver_name:   str
    resolver_ip:     str
    hostname:        str
    dns_time_ms:     Optional[float]   # None if lookup failed
    tcp_time_ms:     Optional[float]   # None if connection failed
    http_time_ms:    Optional[float]   # None if HTTP request failed
    total_time_ms:   Optional[float]   # sum of successful components
    dns_error:       str
    tcp_error:       str
    http_error:      str
    resolved_ip:     str               # IP address returned by DNS


# ──────────────────────────────────────────────
# MEASUREMENT FUNCTIONS
# ──────────────────────────────────────────────

def measure_dns(hostname: str, resolver_ip: str) -> tuple[Optional[float], str, str]:
    """
    Measure DNS lookup time using dnspython directed at a specific resolver.

    Returns
    -------
    (elapsed_ms, resolved_ip, error_msg)
        elapsed_ms  : lookup time in milliseconds, or None on failure
        resolved_ip : first A-record IP, or '' on failure
        error_msg   : '' on success, description on failure
    """
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = [resolver_ip]
    resolver.timeout = 5
    resolver.lifetime = 5

    t_start = time.perf_counter()
    try:
        answers = resolver.resolve(hostname, "A")
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0
        resolved_ip = str(answers[0])
        return elapsed_ms, resolved_ip, ""
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
    Measure TCP three-way handshake time by timing socket.connect().

    This captures: SYN → SYN-ACK → ACK  (client side)
    Note: socket.connect() returns after the ACK is sent, so we measure
          the network RTT to the server, which is the dominant factor.

    Returns
    -------
    (elapsed_ms, error_msg)
    """
    if not ip:
        return None, "No IP (DNS failed)"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TCP_TIMEOUT_S)
        t_start = time.perf_counter()
        sock.connect((ip, port))
        t_end = time.perf_counter()
        sock.close()
        return (t_end - t_start) * 1000.0, ""
    except socket.timeout:
        return None, "TCP timeout"
    except ConnectionRefusedError:
        return None, "Connection refused"
    except Exception as exc:
        return None, str(exc)


def measure_http_load(hostname: str, resolved_ip: str) -> tuple[Optional[float], str]:
    """
    Measure full HTTP GET time for the root path of a hostname.

    We use requests with a custom Host header so the TCP connection goes
    to the already-resolved IP (avoiding a second DNS lookup inside requests).

    Returns
    -------
    (elapsed_ms, error_msg)
        elapsed_ms : time from request start to first byte of response body
    """
    if not resolved_ip:
        return None, "No IP (DNS failed)"
    url = f"http://{hostname}/"
    headers = {"Host": hostname}
    try:
        t_start = time.perf_counter()
        # stream=True so we measure time-to-first-byte
        with requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S,
                          stream=True, allow_redirects=True) as resp:
            # consume first chunk to trigger TTFB
            next(resp.iter_content(chunk_size=1024), None)
        t_end = time.perf_counter()
        return (t_end - t_start) * 1000.0, ""
    except requests.exceptions.Timeout:
        return None, "HTTP timeout"
    except requests.exceptions.ConnectionError as exc:
        return None, f"Connection error: {exc}"
    except Exception as exc:
        return None, str(exc)


# ──────────────────────────────────────────────
# CORE MEASUREMENT LOOP
# ──────────────────────────────────────────────

def run_measurements() -> list[Measurement]:
    """
    Run all (trial × resolver × hostname) combinations.

    Outer loop: resolvers  (keeps DNS cache effects consistent per resolver)
    Middle loop: hostnames
    Inner loop: trials

    This ordering means each resolver's cache warms up naturally across
    hostnames, which reflects real-world browsing patterns.
    """
    results: list[Measurement] = []
    total = len(RESOLVERS) * len(TARGET_HOSTS) * N_TRIALS
    count = 0

    for resolver_name, resolver_ip in RESOLVERS.items():
        print(f"\n{'='*60}")
        print(f"Resolver: {resolver_name} ({resolver_ip})")
        print(f"{'='*60}")

        for hostname in TARGET_HOSTS:
            print(f"  Host: {hostname}")

            for trial in range(1, N_TRIALS + 1):
                count += 1
                pct = count / total * 100
                print(f"    Trial {trial:2d}/{N_TRIALS}  [{pct:5.1f}%]", end="  ")

                # ① DNS timing
                dns_ms, resolved_ip, dns_err = measure_dns(hostname, resolver_ip)

                # ② TCP handshake (uses the IP returned by DNS above)
                if resolved_ip:
                    tcp_ms, tcp_err = measure_tcp_handshake(resolved_ip, TCP_PORT)
                else:
                    tcp_ms, tcp_err = None, "Skipped (no IP)"

                # ③ HTTP time-to-first-byte
                if resolved_ip:
                    http_ms, http_err = measure_http_load(hostname, resolved_ip)
                else:
                    http_ms, http_err = None, "Skipped (no IP)"

                # ④ Component-wise total
                components = [v for v in [dns_ms, tcp_ms] if v is not None]
                total_ms = sum(components) if components else None

                # Status summary
                dns_status  = f"{dns_ms:.1f}ms"  if dns_ms  else f"FAIL({dns_err})"
                tcp_status  = f"{tcp_ms:.1f}ms"  if tcp_ms  else f"FAIL({tcp_err})"
                http_status = f"{http_ms:.1f}ms" if http_ms else f"FAIL({http_err})"
                print(f"DNS={dns_status}  TCP={tcp_status}  HTTP={http_status}")

                results.append(Measurement(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    trial=trial,
                    resolver_name=resolver_name,
                    resolver_ip=resolver_ip,
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
# OUTPUT — CSV WRITERS
# ──────────────────────────────────────────────

def save_raw(results: list[Measurement], path: str) -> None:
    """Write one row per measurement to CSV."""
    fieldnames = [f.name for f in fields(Measurement)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in results:
            writer.writerow({f.name: getattr(m, f.name) for f in fields(m)})
    print(f"\nRaw data saved → {path}  ({len(results)} rows)")


def save_summary(results: list[Measurement], path: str) -> None:
    """
    Compute per-(resolver, hostname) summary statistics and write to CSV.

    Statistics computed
    -------------------
        n          : number of successful trials
        mean       : arithmetic mean (ms)
        std        : sample standard deviation (ms)
        min        : minimum observed value (ms)
        p25        : 25th percentile
        median     : 50th percentile
        p75        : 75th percentile
        p95        : 95th percentile
        max        : maximum observed value (ms)
    """
    # Group results
    groups: dict[tuple, list] = {}
    for m in results:
        key = (m.resolver_name, m.hostname)
        groups.setdefault(key, []).append(m)

    def pct(data: list[float], p: float) -> float:
        """Compute the p-th percentile of a sorted list."""
        if not data:
            return float("nan")
        data_sorted = sorted(data)
        k = (len(data_sorted) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
        return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (k - lo)

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

    rows = []
    for (resolver, host), measurements in sorted(groups.items()):
        dns_vals  = [m.dns_time_ms  for m in measurements if m.dns_time_ms  is not None]
        tcp_vals  = [m.tcp_time_ms  for m in measurements if m.tcp_time_ms  is not None]
        http_vals = [m.http_time_ms for m in measurements if m.http_time_ms is not None]

        row = {"resolver": resolver, "hostname": host,
               "dns_success_rate":  len(dns_vals)  / len(measurements),
               "tcp_success_rate":  len(tcp_vals)  / len(measurements),
               "http_success_rate": len(http_vals) / len(measurements)}
        for prefix, vals in [("dns", dns_vals), ("tcp", tcp_vals), ("http", http_vals)]:
            for k, v in stats_for(vals).items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)

    if rows:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Summary stats saved → {path}  ({len(rows)} groups)")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("DCIT 417 — DNS Latency Measurement Tool")
    print(f"Resolvers : {list(RESOLVERS.keys())}")
    print(f"Hosts     : {TARGET_HOSTS}")
    print(f"Trials    : {N_TRIALS} per (resolver × host) = "
          f"{len(RESOLVERS) * len(TARGET_HOSTS) * N_TRIALS} total measurements")
    print(f"Est. time : ~{len(RESOLVERS)*len(TARGET_HOSTS)*N_TRIALS*INTER_TRIAL_DELAY/60:.1f} min")

    results = run_measurements()

    raw_path     = os.path.join(OUTPUT_DIR, "raw_measurements.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")

    save_raw(results, raw_path)
    save_summary(results, summary_path)

    print("\nDone. Next step: run analysis.py")