"""Reproduce the static isolated-entangler baseline comparison table."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import qutip as qt
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsys_audit.physics_engine import PhysicsEngine
from qsys_audit.workload_replay import WorkloadReplay


def load_config() -> Dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def proj(psi: qt.Qobj) -> qt.Qobj:
    return psi * psi.dag()


def cnot_fidelity_vector(physics: PhysicsEngine, stress_name: str) -> Tuple[float, float]:
    stress = physics.stress_profiles()[stress_name]
    cnot_cfg = physics.config.get("cnot", {})

    zero = physics.zero
    one = physics.one
    plus = (zero + one).unit()

    u_cnot = qt.Qobj(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        ),
        dims=[[2, 2], [2, 2]],
    )

    inputs = [
        qt.tensor(zero, zero),
        qt.tensor(one, zero),
        qt.tensor(plus, zero),
        qt.tensor(zero, plus),
    ]

    vals = []
    for psi_in in inputs:
        rho_init = qt.tensor(proj(zero), proj(psi_in))
        rho_fin = physics.evolve_echoed_cr_cnot(
            rho_init_3q=rho_init,
            stress=stress,
            theta_zx=float(cnot_cfg.get("theta_zx", 0.9)),
            rx_target_corr=float(cnot_cfg.get("rx_target_corr", -np.pi / 2.0)),
            rz_control_corr=float(cnot_cfg.get("rz_control_corr", -1.6408)),
            beta_ix=float(cnot_cfg.get("beta_ix", 0.2)),
            gamma_zi=float(cnot_cfg.get("gamma_zi", 0.05)),
            sigma=float(cnot_cfg.get("sigma", 0.25)),
            t_duration=float(cnot_cfg.get("duration", 1.5)),
            nsteps=int(cnot_cfg.get("nsteps", 120)),
        )

        rho_q2 = rho_fin.ptrace([1, 2])
        psi_ideal = (u_cnot * psi_in).unit()
        f_val = float((proj(psi_ideal) * rho_q2).tr().real)
        vals.append(f_val)

    vals_np = np.asarray(vals, dtype=float)
    return float(np.mean(vals_np)), float(np.min(vals_np))


def run() -> None:
    cfg = load_config()
    physics = PhysicsEngine(cfg)
    replay = WorkloadReplay(physics, cfg)

    iso_expected, iso_worst = cnot_fidelity_vector(physics, "aggressive")

    workloads = [
        ("ghz", "GHZ"),
        ("w_state", "W-state"),
        ("qft", "QFT"),
        ("vqe_su2", "VQE-SU2"),
        ("vqe_two_local", "VQE-TwoLocal"),
    ]

    aggressive = physics.stress_profiles()["aggressive"]
    rows = []

    for workload_key, display_name in workloads:
        metric = replay.benchmark_workload(workload_key, aggressive)

        static_expected = iso_expected ** metric.n_cx
        static_worst = iso_worst ** metric.n_cx

        rows.append(
            {
                "Workload": display_name,
                "CX_count": metric.n_cx,
                "Static_expected": static_expected,
                "Actual_expected": metric.f_expected,
                "Gap_expected": static_expected - metric.f_expected,
                "Static_worst": static_worst,
                "Actual_worst": metric.f_worst,
                "Gap_worst": static_worst - metric.f_worst,
            }
        )

    df = pd.DataFrame(rows)
    print("We compare isolated-entangler static predictions against aggressive workload replay results.")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
