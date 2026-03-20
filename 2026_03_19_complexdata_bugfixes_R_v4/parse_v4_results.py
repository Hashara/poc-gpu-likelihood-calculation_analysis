#!/usr/bin/env python3
"""
Parse IQ-TREE3 v4 bugfix results: extract log-likelihoods and rate parameters,
compare VANILA vs OPENACC builds across all models, tree types, and trees.
"""

import os
import re
import csv
from collections import defaultdict

RESULTS_DIR = "/Users/u7826985/Projects/Nvidia/results/2026_03_19_complexdata_bugfixes_R_v4/"
SCRIPT_DIR = "/Users/u7826985/Projects/Nvidia/poc-gpu-likelihood-calculation_analysis/2026_03_19_complexdata_bugfixes_R_v4/"
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "v4_results.csv")


def parse_iqtree_file(filepath):
    """Extract log-likelihood and rate info from a .iqtree file."""
    log_likelihood = None
    site_proportion_rates = ""
    rates_info = ""

    with open(filepath, "r") as f:
        content = f.read()

    # Extract log-likelihood
    m = re.search(r"Log-likelihood of the tree:\s+([-\d.]+)", content)
    if m:
        log_likelihood = float(m.group(1))

    # Extract site proportion and rates (for +R4 models)
    m = re.search(r"Site proportion and rates:\s+(.+)", content)
    if m:
        site_proportion_rates = m.group(1).strip()

    # Extract category table for +R4
    m = re.search(
        r"Category\s+Relative_rate\s+Proportion\n((?:\s+\d+\s+[\d.]+\s+[\d.]+\n)+)",
        content,
    )
    if m:
        rates_info = m.group(1).strip().replace("\n", " | ")

    combined_rates = ""
    if site_proportion_rates:
        combined_rates = site_proportion_rates
    if rates_info:
        combined_rates += " [" + rates_info + "]" if combined_rates else rates_info

    return log_likelihood, combined_rates


def parse_filename(filename):
    """Extract build type, verbose flag from filename."""
    is_verbose = "verbose" in filename
    if "OPENACC" in filename:
        build = "OPENACC"
    elif "VANILA" in filename:
        build = "VANILA"
    else:
        build = "UNKNOWN"
    return build, is_verbose


def main():
    rows = []

    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in files:
            if not fname.endswith(".iqtree"):
                continue

            filepath = os.path.join(root, fname)

            # Extract metadata from directory structure:
            # .../data_type/tree_type/model/tree_N/filename.iqtree
            rel = os.path.relpath(filepath, RESULTS_DIR)
            parts = rel.split(os.sep)
            if len(parts) < 5:
                continue

            data_type = parts[0]   # DNA or AA
            tree_type = parts[1]   # rooted or unrooted
            model = parts[2]       # GTR, GTR+G4, LG+R4, etc.
            tree_num = parts[3]    # tree_1 .. tree_10

            build, is_verbose = parse_filename(fname)
            log_likelihood, rates_info = parse_iqtree_file(filepath)

            rows.append({
                "data_type": data_type,
                "tree_type": tree_type,
                "model": model,
                "tree_num": tree_num,
                "build": build,
                "is_verbose": is_verbose,
                "log_likelihood": log_likelihood,
                "rates_info": rates_info,
            })

    # Sort rows for consistent output
    rows.sort(key=lambda r: (
        r["data_type"], r["tree_type"], r["model"], r["tree_num"],
        r["build"], r["is_verbose"]
    ))

    # Write CSV
    fieldnames = [
        "data_type", "tree_type", "model", "tree_num", "build",
        "is_verbose", "log_likelihood", "rates_info",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}\n")

    # ── Summary: compare VANILA vs OPENACC ──────────────────────────────
    # Group by (data_type, tree_type, model, is_verbose)
    # For each group, match VANILA and OPENACC by tree_num

    groups = defaultdict(lambda: {"VANILA": {}, "OPENACC": {}})
    for r in rows:
        key = (r["data_type"], r["tree_type"], r["model"], r["is_verbose"])
        groups[key][r["build"]][r["tree_num"]] = r

    print("=" * 120)
    print(f"{'Data':>4s}  {'Tree':>8s}  {'Model':<12s} {'Verbose':>7s}  "
          f"{'#Trees':>6s}  {'MeanDiff':>12s}  {'MaxDiff':>12s}  {'Flag':>6s}  Per-tree diffs")
    print("=" * 120)

    flagged_groups = []

    for key in sorted(groups.keys()):
        data_type, tree_type, model, is_verbose = key
        vanila = groups[key]["VANILA"]
        openacc = groups[key]["OPENACC"]

        common_trees = sorted(set(vanila.keys()) & set(openacc.keys()),
                              key=lambda t: int(t.split("_")[1]))

        if not common_trees:
            print(f"{data_type:>4s}  {tree_type:>8s}  {model:<12s} {str(is_verbose):>7s}  "
                  f"{'0':>6s}  {'N/A':>12s}  {'N/A':>12s}  {'':>6s}")
            continue

        diffs = []
        per_tree_details = []
        for t in common_trees:
            v_ll = vanila[t]["log_likelihood"]
            o_ll = openacc[t]["log_likelihood"]
            if v_ll is not None and o_ll is not None:
                d = o_ll - v_ll
                diffs.append(d)
                per_tree_details.append(f"{t}:{d:+.6e}")
            else:
                per_tree_details.append(f"{t}:MISSING")

        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs, key=abs)
            flag = "***" if abs(max_diff) > 1e-6 else ""
        else:
            mean_diff = float("nan")
            max_diff = float("nan")
            flag = ""

        verbose_str = "yes" if is_verbose else "no"

        # For +R4 models, always show per-tree details; for others, only if flagged
        show_model = model.endswith("+R4") or model.endswith("+C60")
        detail_str = "  ".join(per_tree_details) if (flag or show_model) else ""

        line = (f"{data_type:>4s}  {tree_type:>8s}  {model:<12s} {verbose_str:>7s}  "
                f"{len(common_trees):>6d}  {mean_diff:>+12.6e}  {max_diff:>+12.6e}  {flag:>6s}")
        print(line)
        if detail_str:
            print(f"{'':>60s}{detail_str}")

        if flag:
            flagged_groups.append((data_type, tree_type, model, is_verbose, max_diff))

    print("=" * 120)

    # Print flagged summary
    if flagged_groups:
        print(f"\n*** FLAGGED: {len(flagged_groups)} group(s) with |diff| > 1e-6 ***")
        for data_type, tree_type, model, is_verbose, max_diff in flagged_groups:
            v = " (verbose)" if is_verbose else ""
            print(f"  {data_type}/{tree_type}/{model}{v}: max diff = {max_diff:+.6e}")
    else:
        print("\nAll VANILA vs OPENACC differences are within 1e-6. No flags.")

    # ── Also print +R4 rates comparison ─────────────────────────────────
    r4_groups = {k: v for k, v in groups.items() if "+R4" in k[2]}
    if r4_groups:
        print("\n" + "=" * 120)
        print("+R4 RATE PARAMETER COMPARISON")
        print("=" * 120)
        for key in sorted(r4_groups.keys()):
            data_type, tree_type, model, is_verbose = key
            vanila = r4_groups[key]["VANILA"]
            openacc = r4_groups[key]["OPENACC"]
            verbose_str = " (verbose)" if is_verbose else ""
            print(f"\n--- {data_type}/{tree_type}/{model}{verbose_str} ---")
            common_trees = sorted(set(vanila.keys()) & set(openacc.keys()),
                                  key=lambda t: int(t.split("_")[1]))
            for t in common_trees:
                v_rates = vanila[t].get("rates_info", "")
                o_rates = openacc[t].get("rates_info", "")
                match = "MATCH" if v_rates == o_rates else "DIFFER"
                print(f"  {t}: {match}")
                if match == "DIFFER":
                    print(f"    VANILA:  {v_rates}")
                    print(f"    OPENACC: {o_rates}")


if __name__ == "__main__":
    main()
