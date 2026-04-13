"""Command-line entry point for qsys_audit.

Author: Anonymous Authors
Institution: Blinded for Review
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import yaml

# Ensure src-layout imports work when running: python audit_cli.py
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline cross-layer workload auditor for quantum operating systems and schedulers."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--workload",
        required=True,
        help="Workload alias (example: w_state, ghz, qft, vqe_two_local).",
    )
    parser.add_argument(
        "--stress",
        required=True,
        choices=["calibrated", "moderate", "aggressive"],
        help="Stress profile for concurrent hardware interference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of text.",
    )
    return parser


def classify_verdict(d_val: float, dlog: float, dinc: float, cfg: Dict) -> str:
    thresholds = cfg.get("verification_thresholds", {})
    if (
        d_val <= thresholds.get("eps_d_pass", 0.02)
        and dlog <= thresholds.get("eps_log_pass", 0.05)
        and dinc <= thresholds.get("eps_inc_pass", 0.05)
    ):
        return "PASS"

    if (
        d_val >= thresholds.get("d_hard", 0.30)
        or dlog >= thresholds.get("log_hard", 0.90)
        or dinc >= thresholds.get("inc_hard", 0.50)
    ):
        return "HARD FAIL"

    return "SOFT FAIL"


def run_cli(args: argparse.Namespace) -> Dict:
    # Defer heavy scientific imports so --help remains available even before dependency install.
    from qsys_audit.physics_engine import PhysicsEngine
    from qsys_audit.reconstruction import ReconstructionEngine
    from qsys_audit.workload_replay import WorkloadReplay

    cfg = load_config(args.config)
    physics = PhysicsEngine(cfg)
    recon = ReconstructionEngine(physics, cfg)
    replay = WorkloadReplay(physics, cfg)

    profiles = physics.stress_profiles()
    if args.stress not in profiles:
        raise ValueError(f"Unknown stress profile: {args.stress}")

    metrics = replay.benchmark_workload(args.workload, profiles[args.stress])

    # Single-qubit channel reconstruction summary is included for full cross-layer reporting.
    theta_gate = float(cfg.get("single_qubit_gate", {}).get("theta_gate", 1.5707963267948966))
    c_phys = recon.perform_qpt_pipeline(theta_gate, profiles[args.stress])
    c_ideal = recon.ideal_choi_ry(theta_gate)
    d_val, c_proj = recon.process_distance_d_physical(c_ideal, c_phys)
    fit = recon.fit_unitary_plus_noise_from_choi(c_proj)
    dlog = recon.delta_log(physics.ry(theta_gate), fit["U_eff"])
    dinc = recon.delta_inc(fit["pauli_probs"])

    verdict = classify_verdict(d_val, dlog, dinc, cfg)

    return {
        "workload": metrics.workload,
        "stress": metrics.stress,
        "expected_fidelity": metrics.f_expected,
        "worst_case_fidelity": metrics.f_worst,
        "fidelity_spread": metrics.f_spread,
        "n_cx": metrics.n_cx,
        "n_1q": metrics.n_1q,
        "depth": metrics.depth,
        "size": metrics.size,
        "process_distance_d": d_val,
        "delta_log": dlog,
        "delta_inc": dinc,
        "verdict": verdict,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_cli(args)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Workload: {result['workload']}")
    print(f"Stress: {result['stress']}")
    print(f"Expected Fidelity: {result['expected_fidelity']:.6f}")
    print(f"Worst-Case Fidelity: {result['worst_case_fidelity']:.6f}")
    print(f"Fidelity Spread: {result['fidelity_spread']:.6f}")
    print(f"Verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
