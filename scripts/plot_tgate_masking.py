"""Reproduce the T-gate multi-basis diagnostic figure and summary table."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsys_audit.physics_engine import PhysicsEngine


plt.style.use("seaborn-v0_8-paper")


def load_config() -> Dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gaussian_theta_to_amplitude(theta: float, sigma: float) -> float:
    return (theta / 2.0) / (sigma * np.sqrt(2.0 * np.pi))


def _gate_axis_op(physics: PhysicsEngine, axis: str) -> qt.Qobj:
    if axis == "x":
        return qt.tensor(physics.I, qt.sigmax(), physics.I)
    if axis == "y":
        return qt.tensor(physics.I, qt.sigmay(), physics.I)
    if axis == "z":
        return qt.tensor(physics.I, qt.sigmaz(), physics.I)
    raise ValueError(f"Unsupported gate axis: {axis}")


def get_final_state_for_gate(physics: PhysicsEngine, rho_q_in: qt.Qobj, gate_spec: Dict, stress_name: str, nsteps: int) -> qt.Qobj:
    stress = physics.stress_profiles()[stress_name]
    rho_init_3q = qt.tensor(physics.zero.proj(), rho_q_in, physics.zero.proj())

    seg_durations = [seg.get("duration", 1.0) for seg in gate_spec.get("segments", [])]
    duration = max([1.0] + seg_durations)

    tlist = np.linspace(0.0, duration, nsteps)
    h_static, h_background = physics._background_terms_3q(stress, duration)
    h_total = [h_static] + h_background

    for seg in gate_spec.get("segments", []):
        sigma = float(seg.get("sigma", 0.2))
        amplitude = _gaussian_theta_to_amplitude(float(seg["theta"]), sigma)
        coeff = physics.pulse_shape(shape="gaussian", A=amplitude, sigma=sigma, t_duration=duration)
        h_total.append([_gate_axis_op(physics, seg["axis"]), coeff])

    sol = qt.mesolve(h_total, rho_init_3q, tlist, c_ops=[], e_ops=[])
    rho_q = sol.states[-1].ptrace(1)

    for u_post in gate_spec.get("virtual_post", []):
        rho_q = u_post * rho_q * u_post.dag()
    return rho_q


def fidelity_pure_to_state(psi_target: qt.Qobj, rho: qt.Qobj) -> float:
    return float(((psi_target * psi_target.dag()) * rho).tr().real)


def run() -> None:
    cfg = load_config()
    physics = PhysicsEngine(cfg)

    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    plus = (zero + one).unit()
    minus = (zero - one).unit()
    plus_i = (zero + 1j * one).unit()
    minus_i = (zero - 1j * one).unit()

    mub_states: List[Tuple[str, qt.Qobj]] = [
        ("0", zero),
        ("1", one),
        ("+", plus),
        ("-", minus),
        ("+i", plus_i),
        ("-i", minus_i),
    ]

    t_gate = {
        "ideal_U": physics.rz(np.pi / 4.0),
        "segments": [],
        "virtual_post": [physics.rz(np.pi / 4.0)],
    }

    stress_order = ["calibrated", "moderate", "aggressive"]
    stress_label = {"calibrated": "Calibrated", "moderate": "Moderate", "aggressive": "Aggressive"}

    rows = []
    for stress in stress_order:
        for label, psi_in in mub_states:
            rho_exact = get_final_state_for_gate(
                physics=physics,
                rho_q_in=psi_in.proj(),
                gate_spec=t_gate,
                stress_name=stress,
                nsteps=50,
            )
            psi_ideal = (t_gate["ideal_U"] * psi_in).unit()
            f_val = fidelity_pure_to_state(psi_ideal, rho_exact)
            rows.append({"Stress": stress_label[stress], "Basis": label, "Fidelity": f_val})

    df = pd.DataFrame(rows)
    gap_summary = (
        df.assign(Gap=lambda x: 1.0 - x["Fidelity"])
        .groupby("Stress", as_index=False)
        .agg(mean_gap=("Gap", "mean"), min_gap=("Gap", "min"))
    )

    print("We evaluate T-gate masking behavior across six mutually unbiased bases.")
    print(gap_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    bases = [b for b, _ in mub_states]
    x = np.arange(len(bases))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for idx, stress in enumerate(["Calibrated", "Moderate", "Aggressive"]):
        sub = df[df["Stress"] == stress].set_index("Basis").loc[bases]
        ax.bar(x + (idx - 1) * width, sub["Fidelity"].to_numpy(), width=width, label=stress)

    ax.set_xticks(x)
    ax.set_xticklabels(bases)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fidelity")
    ax.set_xlabel("Mutually Unbiased Basis Input")
    ax.set_title("We evaluate T-gate fidelity profiles across mutually unbiased bases")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = ROOT / "tgate_masking_fig3.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"We saved the T-gate masking figure to {out_path.name}.")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
