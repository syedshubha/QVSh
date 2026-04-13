"""Reproduce the Two-Qubit Workload Fidelity Spectrum figure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsys_audit.physics_engine import PhysicsEngine
from qsys_audit.workload_replay import WorkloadReplay


plt.style.use("seaborn-v0_8-paper")


def load_config() -> Dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run() -> None:
    cfg = load_config()
    physics = PhysicsEngine(cfg)
    replay = WorkloadReplay(physics, cfg)

    workloads: List[tuple[str, str]] = [
        ("ghz", "GHZ"),
        ("w_state", "W-state"),
        ("qft", "QFT"),
        ("vqe_su2", "VQE-SU2"),
        ("vqe_two_local", "VQE-TwoLocal"),
    ]
    stress_order = ["calibrated", "moderate", "aggressive"]
    stress_label = {"calibrated": "Calibrated", "moderate": "Moderate", "aggressive": "Aggressive"}

    rows = []
    for stress in stress_order:
        profile = physics.stress_profiles()[stress]
        for workload_key, workload_name in workloads:
            metric = replay.benchmark_workload(workload_key, profile)
            rows.append(
                {
                    "Workload": workload_name,
                    "Stress": stress_label[stress],
                    "Expected": metric.f_expected,
                    "Worst": metric.f_worst,
                    "Spread": metric.f_spread,
                }
            )

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    metric_cols = [
        ("Expected", "We evaluate expected workload fidelity"),
        ("Worst", "We evaluate worst-case workload fidelity"),
        ("Spread", "We evaluate workload fidelity spread"),
    ]

    x = np.arange(len(workloads))
    width = 0.24

    for ax, (col, title) in zip(axes, metric_cols):
        for idx, stress in enumerate(["Calibrated", "Moderate", "Aggressive"]):
            sub = df[df["Stress"] == stress].set_index("Workload")
            ordered = [name for _, name in workloads]
            vals = sub.loc[ordered, col].to_numpy()
            ax.bar(x + (idx - 1) * width, vals, width=width, label=stress)

        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(col)
        if col != "Spread":
            ax.set_ylim(0.0, 1.05)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([name for _, name in workloads])
    axes[0].legend(ncol=3, frameon=False, loc="lower left")

    fig.suptitle("We compare the two-qubit workload fidelity spectrum across stress regimes", y=1.01)
    fig.tight_layout()

    out_path = ROOT / "workload_spectrum_fig2.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"We saved the workload fidelity spectrum figure to {out_path.name}.")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
