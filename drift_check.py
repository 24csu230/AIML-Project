#!/usr/bin/env python3
"""
CSL236 - End-Term Project (Bonus Feature)
Automated Input Data Drift Detection Script
Compares production prediction telemetry against training baseline statistics.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "prediction_logs.jsonl")
META_FILE_PATH = os.path.join(BASE_DIR, "model", "baseline_metadata.json")
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "cybersecurity_threats.csv")


def generate_sample_logs_if_empty(log_path: str, count: int = 25):
    """Generate simulated inference logs for testing drift detection."""
    print(f"[*] Simulating {count} production inference log entries...")
    np.random.seed(42)
    sample_entries = []
    countries = ["USA", "India", "UK", "Germany", "France", "China"]
    attack_types = ["Ransomware", "DDoS", "SQL Injection", "Phishing", "Malware"]
    industries = ["Banking", "Healthcare", "IT", "Government", "Retail"]
    sources = ["Nation-state", "Hacker Group", "Insider", "Unknown"]
    vulns = ["Zero-day", "Unpatched Software", "Weak Passwords", "Social Engineering"]
    defenses = ["Firewall", "AI-based Detection", "Encryption", "VPN", "Antivirus"]

    for _ in range(count):
        entry = {
            "timestamp": "2026-11-16T12:00:00Z",
            "input": {
                "country": np.random.choice(countries),
                "year": int(np.random.choice([2023, 2024, 2025])),
                "attack_type": np.random.choice(attack_types),
                "target_industry": np.random.choice(industries),
                "number_of_affected_users": int(np.random.uniform(50000, 900000)),
                "attack_source": np.random.choice(sources),
                "security_vulnerability_type": np.random.choice(vulns),
                "defense_mechanism_used": np.random.choice(defenses),
                "incident_resolution_time": float(round(np.random.uniform(5.0, 65.0), 1)),
            },
            "output": {
                "prediction": "High" if np.random.rand() > 0.5 else "Medium",
                "risk_score": float(round(np.random.uniform(0.4, 0.95), 3)),
            },
        }
        sample_entries.append(entry)

    with open(log_path, "w") as f:
        for item in sample_entries:
            f.write(json.dumps(item) + "\n")
    print(f"[+] Written {count} sample logs to: {log_path}")


def load_baseline():
    """Load baseline statistics from metadata or raw training dataset."""
    if os.path.exists(META_FILE_PATH):
        with open(META_FILE_PATH, "r") as f:
            meta = json.load(f)
            return meta.get("features", {}).get("numerical", {})

    # Compute directly from CSV
    if os.path.exists(DATA_FILE_PATH):
        df = pd.read_csv(DATA_FILE_PATH)
        rename_map = {
            "Number of Affected Users": "number_of_affected_users",
            "Incident Resolution Time (in Hours)": "incident_resolution_time",
            "Year": "year",
        }
        df = df.rename(columns=rename_map)
        baseline = {}
        for col in ["number_of_affected_users", "incident_resolution_time", "year"]:
            baseline[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        return baseline

    raise FileNotFoundError("Neither baseline metadata nor training dataset was found.")


def main():
    parser = argparse.ArgumentParser(description="Check data drift between production logs and training baseline")
    parser.add_argument("--log_path", type=str, default=LOG_FILE_PATH, help="Path to prediction_logs.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5, help="Z-score drift threshold")
    parser.add_argument("--simulate", action="store_true", help="Simulate sample logs if missing")
    parser.add_argument("--output_report", type=str, default="drift_report.json", help="Path to save drift report")

    args = parser.parse_args()

    if not os.path.exists(args.log_path) or os.path.getsize(args.log_path) == 0:
        if args.simulate:
            generate_sample_logs_if_empty(args.log_path, count=30)
        else:
            print(f"[!] Log file {args.log_path} is empty or missing. Use --simulate to generate mock production logs.")
            sys.exit(0)

    # Read logged inputs
    logged_records = []
    with open(args.log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    payload = json.loads(line)
                    if "input" in payload:
                        logged_records.append(payload["input"])
                except json.JSONDecodeError:
                    continue

    if len(logged_records) == 0:
        print("[!] No valid input logs found for drift analysis.")
        sys.exit(0)

    df_logs = pd.DataFrame(logged_records)
    baseline_stats = load_baseline()

    print("\n" + "=" * 70)
    print(" CSL236 DATA DRIFT MONITORING REPORT (PRODUCTION vs BASELINE)")
    print("=" * 70)
    print(f"Logged Inference Samples: {len(df_logs)}")
    print(f"Z-Score Drift Alarm Threshold: {args.threshold} std deviations\n")

    drift_results = {}
    any_drift_detected = False

    header = f"{'Feature':<28} | {'Train Mean':<11} | {'Log Mean':<11} | {'Z-Shift':<8} | {'Status'}"
    print(header)
    print("-" * len(header))

    for feature, stats in baseline_stats.items():
        if feature not in df_logs.columns:
            continue

        train_mean = stats["mean"]
        train_std = stats["std"]
        log_mean = float(df_logs[feature].mean())
        log_std = float(df_logs[feature].std()) if len(df_logs) > 1 else 0.0

        z_shift = abs(log_mean - train_mean) / (train_std if train_std != 0 else 1.0)
        is_drift = z_shift > args.threshold

        if is_drift:
            any_drift_detected = True
            status_str = "⚠️ DRIFT DETECTED"
        else:
            status_str = "✅ STABLE"

        drift_results[feature] = {
            "baseline_mean": round(train_mean, 2),
            "baseline_std": round(train_std, 2),
            "production_mean": round(log_mean, 2),
            "production_std": round(log_std, 2),
            "z_shift": round(z_shift, 4),
            "drift_detected": is_drift,
        }

        print(f"{feature:<28} | {train_mean:<11.2f} | {log_mean:<11.2f} | {z_shift:<8.3f} | {status_str}")

    print("=" * 70)
    if any_drift_detected:
        print("[!] ALERT: Feature drift detected! Recommend scheduling model retraining pipeline.")
    else:
        print("[+] SUCCESS: All features are within acceptable distribution bounds.")

    with open(args.output_report, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"[+] Drift analysis report saved to: {args.output_report}\n")


if __name__ == "__main__":
    main()
