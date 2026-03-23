"""
DCIT 417 — Network Modelling & Performance Analysis
Project: DNS Latency and Its Impact on Web Page Load Time

Phase 3 — Statistical Analysis
================================
Loads raw_measurements.csv (produced by dns_measure.py) and performs:

    1. Descriptive statistics per (resolver, hostname)
    2. Distribution fitting  — tests Exponential and Normal fits
    3. Burstiness analysis   — coefficient of variation (CoV), index of dispersion
    4. CDF / ECDF plots      — visual comparison across resolvers
    5. Box plots             — DNS and TCP latency by resolver
    6. Bar charts            — mean DNS latency with 95% CI error bars
    7. Heatmap               — median DNS latency (resolver × hostname)

Outputs
-------
    plots/dns_boxplot.png
    plots/dns_cdf.png
    plots/dns_mean_bar.png
    plots/dns_heatmap.png
    plots/tcp_boxplot.png
    results/distribution_fits.csv   — GoF test results per group

Usage
-----
    pip install numpy pandas matplotlib scipy seaborn
    python analysis.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

RAW_CSV    = "results/raw_measurements.csv"
PLOTS_DIR  = "plots"
RESULTS_DIR= "results"

# Colour palette — one colour per resolver (colourblind-friendly)
RESOLVER_COLOURS = {
    "google_public":     "#2563EB",   # blue
    "cloudflare_public": "#16A34A",   # green
    "isp":               "#D97706",   # amber
    "campus":            "#DC2626",   # red
}
RESOLVER_LABELS = {
    "google_public":     "Google (8.8.8.8)",
    "cloudflare_public": "Cloudflare (1.1.1.1)",
    "isp":               "ISP resolver",
    "campus":            "Campus DNS",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family":     "DejaVu Sans",
    "font.size":       10,
})

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and do basic cleaning."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {list(df.columns)}")
    # Keep only successful DNS measurements for latency analysis
    df["dns_ok"]  = df["dns_time_ms"].notna()
    df["tcp_ok"]  = df["tcp_time_ms"].notna()
    df["http_ok"] = df["http_time_ms"].notna()
    return df


# ──────────────────────────────────────────────
# 1. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-resolver summary statistics for DNS lookup time.

    Key metrics
    -----------
        mean    : average latency (ms)
        std     : spread / variability
        CoV     : coefficient of variation = std/mean  (dimensionless burstiness)
        median  : robust central tendency
        p95     : 95th percentile (tail latency — what 5% of users experience)
    """
    dns_df = df[df["dns_ok"]].copy()

    agg = dns_df.groupby("resolver_name")["dns_time_ms"].agg(
        n="count",
        mean="mean",
        std="std",
        min="min",
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75),
        p95=lambda x: x.quantile(0.95),
        max="max",
    ).reset_index()

    agg["CoV"] = agg["std"] / agg["mean"]   # coefficient of variation
    agg = agg.round(3)

    print("\n── Descriptive Statistics: DNS Lookup Time (ms) ──")
    print(agg.to_string(index=False))
    return agg


# ──────────────────────────────────────────────
# 2. DISTRIBUTION FITTING
# ──────────────────────────────────────────────

def fit_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Exponential and Normal distributions to DNS latency per resolver.

    Why these two?
    --------------
    • Exponential  : theoretically expected for memoryless arrival/service
                     processes (basis of M/M/1 queue in Phase 4)
    • Normal       : often a good empirical fit after the minimum floor

    Goodness-of-fit uses the Kolmogorov-Smirnov (KS) test:
        H0: the data follows the fitted distribution
        small p-value → reject H0 → poor fit

    We also compute log-likelihood to compare fits on the same data.
    """
    dns_df = df[df["dns_ok"]].copy()
    rows = []

    for resolver, group in dns_df.groupby("resolver_name"):
        data = group["dns_time_ms"].values

        # ── Exponential fit
        # MLE for exponential: lambda_hat = 1/mean
        # scipy parameterises as scale = 1/lambda = mean
        loc_exp, scale_exp = stats.expon.fit(data, floc=0)
        ks_exp_stat, ks_exp_p = stats.kstest(data, "expon",
                                              args=(loc_exp, scale_exp))
        ll_exp = np.sum(stats.expon.logpdf(data, loc=loc_exp, scale=scale_exp))

        # ── Normal fit
        loc_norm, scale_norm = stats.norm.fit(data)
        ks_norm_stat, ks_norm_p = stats.kstest(data, "norm",
                                                args=(loc_norm, scale_norm))
        ll_norm = np.sum(stats.norm.logpdf(data, loc=loc_norm, scale=scale_norm))

        # ── Log-normal fit (often a better fit for network latency)
        shape_lnorm, loc_lnorm, scale_lnorm = stats.lognorm.fit(data, floc=0)
        ks_lnorm_stat, ks_lnorm_p = stats.kstest(data, "lognorm",
                                                   args=(shape_lnorm, loc_lnorm, scale_lnorm))
        ll_lnorm = np.sum(stats.lognorm.logpdf(data,
                           shape_lnorm, loc=loc_lnorm, scale=scale_lnorm))

        best = max([("exponential", ll_exp),
                    ("normal",      ll_norm),
                    ("lognormal",   ll_lnorm)],
                   key=lambda x: x[1])[0]

        rows.append({
            "resolver":          resolver,
            "n":                 len(data),
            "mean_ms":           round(data.mean(), 3),
            "std_ms":            round(data.std(), 3),
            # Exponential
            "exp_rate_1_per_ms": round(1 / scale_exp, 6),
            "exp_ks_stat":       round(ks_exp_stat, 4),
            "exp_ks_p":          round(ks_exp_p, 4),
            "exp_log_likelihood":round(ll_exp, 2),
            # Normal
            "norm_mu_ms":        round(loc_norm, 3),
            "norm_sigma_ms":     round(scale_norm, 3),
            "norm_ks_stat":      round(ks_norm_stat, 4),
            "norm_ks_p":         round(ks_norm_p, 4),
            "norm_log_likelihood":round(ll_norm, 2),
            # Log-normal
            "lnorm_shape":       round(shape_lnorm, 4),
            "lnorm_scale":       round(scale_lnorm, 3),
            "lnorm_ks_stat":     round(ks_lnorm_stat, 4),
            "lnorm_ks_p":        round(ks_lnorm_p, 4),
            "lnorm_log_likelihood": round(ll_lnorm, 2),
            "best_fit":          best,
        })

    fit_df = pd.DataFrame(rows)
    print("\n── Distribution Fitting Results ──")
    cols = ["resolver", "mean_ms", "exp_ks_p", "norm_ks_p", "lnorm_ks_p", "best_fit"]
    print(fit_df[cols].to_string(index=False))

    out = os.path.join(RESULTS_DIR, "distribution_fits.csv")
    fit_df.to_csv(out, index=False)
    print(f"Full fit results saved → {out}")
    return fit_df


# ──────────────────────────────────────────────
# 3. BURSTINESS ANALYSIS
# ──────────────────────────────────────────────

def burstiness_analysis(df: pd.DataFrame) -> None:
    """
    Quantify burstiness using coefficient of variation and index of dispersion.

    CoV = std / mean
        CoV < 1  → sub-exponential (less bursty than Poisson)
        CoV = 1  → exponential (Poisson-like, M/M/1 assumption holds)
        CoV > 1  → super-exponential / bursty → need more complex model

    Index of Dispersion (ID) = variance / mean
        ID ≈ 1  → Poisson-like
        ID >> 1 → overdispersed / bursty traffic
    """
    dns_df = df[df["dns_ok"]].copy()
    print("\n── Burstiness Analysis ──")
    print(f"{'Resolver':<25} {'Mean':>8} {'Std':>8} {'CoV':>6} {'ID':>8}  Interpretation")
    print("-" * 75)

    for resolver, group in dns_df.groupby("resolver_name"):
        data = group["dns_time_ms"].values
        mean = data.mean()
        std  = data.std()
        var  = data.var()
        cov  = std / mean
        id_  = var / mean

        if cov < 0.8:
            interp = "sub-exponential (well-behaved)"
        elif cov < 1.2:
            interp = "≈ exponential (M/M/1 suitable)"
        else:
            interp = "super-exponential (bursty!)"

        label = RESOLVER_LABELS.get(resolver, resolver)
        print(f"{label:<25} {mean:>8.2f} {std:>8.2f} {cov:>6.3f} {id_:>8.2f}  {interp}")


# ──────────────────────────────────────────────
# 4. PLOTS
# ──────────────────────────────────────────────

def plot_dns_boxplot(df: pd.DataFrame) -> None:
    """
    Box plot of DNS lookup time per resolver.
    Shows median, IQR, whiskers (1.5×IQR), and outliers.
    """
    dns_df = df[df["dns_ok"]].copy()
    dns_df["resolver_label"] = dns_df["resolver_name"].map(RESOLVER_LABELS)

    order = [RESOLVER_LABELS[r] for r in RESOLVER_COLOURS if r in RESOLVER_LABELS]
    colours = [RESOLVER_COLOURS[r] for r in RESOLVER_COLOURS]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=dns_df, x="resolver_label", y="dns_time_ms",
        order=order, palette=colours,
        fliersize=3, linewidth=0.8, ax=ax
    )
    ax.set_xlabel("DNS Resolver", labelpad=8)
    ax.set_ylabel("DNS Lookup Time (ms)")
    ax.set_title("DNS Lookup Latency Distribution by Resolver", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f ms"))

    # Annotate medians
    for i, resolver_name in enumerate(RESOLVER_COLOURS.keys()):
        group = dns_df[dns_df["resolver_name"] == resolver_name]["dns_time_ms"]
        if len(group) == 0:
            continue
        med = group.median()
        ax.text(i, med + 1, f"{med:.1f}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dns_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_dns_cdf(df: pd.DataFrame) -> None:
    """
    Empirical CDF (ECDF) of DNS lookup time per resolver.

    The ECDF at point x = fraction of measurements ≤ x.
    Useful for reading off tail latency: e.g. what % of lookups complete
    within 50ms? Where does each resolver's 95th percentile sit?
    """
    dns_df = df[df["dns_ok"]].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    for resolver, colour in RESOLVER_COLOURS.items():
        group = dns_df[dns_df["resolver_name"] == resolver]["dns_time_ms"].values
        if len(group) == 0:
            continue
        sorted_data = np.sort(group)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        label = RESOLVER_LABELS.get(resolver, resolver)
        ax.step(sorted_data, cdf, where="post", color=colour,
                linewidth=2, label=label)

    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
               label="95th percentile line")
    ax.set_xlabel("DNS Lookup Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("ECDF of DNS Lookup Latency by Resolver", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dns_cdf.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_dns_mean_bar(df: pd.DataFrame) -> None:
    """
    Bar chart of mean DNS latency per resolver with 95% confidence intervals.

    95% CI = mean ± 1.96 × (std / √n)   [large-sample normal approximation]

    This is your primary comparison chart for the report.
    """
    dns_df = df[df["dns_ok"]].copy()

    agg = dns_df.groupby("resolver_name")["dns_time_ms"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
    agg["label"] = agg["resolver_name"].map(RESOLVER_LABELS)
    agg["colour"] = agg["resolver_name"].map(RESOLVER_COLOURS)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        agg["label"], agg["mean"],
        color=agg["colour"], width=0.5,
        yerr=agg["ci95"], capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5}
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["ci95"] + 0.5,
                f"{row['mean']:.1f} ms",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean DNS Lookup Time (ms)")
    ax.set_title("Mean DNS Latency with 95% Confidence Intervals", fontweight="bold")
    ax.set_xlabel("Resolver")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dns_mean_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_dns_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap: median DNS latency (rows = resolvers, columns = hostnames).

    Reveals which resolver is fastest for which site — useful for
    discussing CDN and anycast effects in the report.
    """
    dns_df = df[df["dns_ok"]].copy()

    pivot = dns_df.pivot_table(
        values="dns_time_ms",
        index="resolver_name",
        columns="hostname",
        aggfunc="median"
    )
    pivot.index = [RESOLVER_LABELS.get(r, r) for r in pivot.index]

    # Shorten hostnames for display
    pivot.columns = [h.replace("www.", "") for h in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd", annot=True, fmt=".0f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Median DNS latency (ms)"}
    )
    ax.set_title("Median DNS Latency Heatmap (Resolver × Hostname)", fontweight="bold")
    ax.set_xlabel("Hostname")
    ax.set_ylabel("Resolver")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "dns_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_tcp_boxplot(df: pd.DataFrame) -> None:
    """TCP handshake time box plot — same structure as DNS box plot."""
    tcp_df = df[df["tcp_ok"]].copy()
    tcp_df["resolver_label"] = tcp_df["resolver_name"].map(RESOLVER_LABELS)

    order = [RESOLVER_LABELS[r] for r in RESOLVER_COLOURS if r in RESOLVER_LABELS]
    colours = list(RESOLVER_COLOURS.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=tcp_df, x="resolver_label", y="tcp_time_ms",
        order=order, palette=colours,
        fliersize=3, linewidth=0.8, ax=ax
    )
    ax.set_xlabel("DNS Resolver used for lookup")
    ax.set_ylabel("TCP Handshake Time (ms)")
    ax.set_title("TCP Handshake Latency by Resolver", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "tcp_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_component_stacked(df: pd.DataFrame) -> None:
    """
    Stacked bar chart showing mean DNS + mean TCP per resolver.
    Visualises the component-wise delay model:  T_total = T_dns + T_tcp
    """
    ok_df = df[df["dns_ok"] & df["tcp_ok"]].copy()

    agg = ok_df.groupby("resolver_name").agg(
        dns_mean=("dns_time_ms", "mean"),
        tcp_mean=("tcp_time_ms", "mean"),
    ).reset_index()
    agg["label"] = agg["resolver_name"].map(RESOLVER_LABELS)

    x = np.arange(len(agg))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    dns_bars = ax.bar(x, agg["dns_mean"], width, label="DNS lookup",
                      color="#2563EB", alpha=0.85)
    tcp_bars = ax.bar(x, agg["tcp_mean"], width, bottom=agg["dns_mean"],
                      label="TCP handshake", color="#D97706", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["label"], rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Component-wise Delay Model: DNS + TCP per Resolver", fontweight="bold")
    ax.legend()

    # Annotate totals
    for i, row in agg.iterrows():
        total = row["dns_mean"] + row["tcp_mean"]
        ax.text(i, total + 0.5, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "component_stacked.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(RAW_CSV)

    print(f"\nSuccess rates:")
    print(f"  DNS  successful: {df['dns_ok'].sum()} / {len(df)}")
    print(f"  TCP  successful: {df['tcp_ok'].sum()} / {len(df)}")
    print(f"  HTTP successful: {df['http_ok'].sum()} / {len(df)}")

    desc = descriptive_stats(df)
    fit_df = fit_distributions(df)
    burstiness_analysis(df)

    print("\n── Generating plots ──")
    plot_dns_boxplot(df)
    plot_dns_cdf(df)
    plot_dns_mean_bar(df)
    plot_dns_heatmap(df)
    plot_tcp_boxplot(df)
    plot_component_stacked(df)

    print("\nAll done. Check plots/ and results/ directories.")